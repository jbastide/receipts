import json

import httpx
import pytest

from receipts import cli
from receipts import support as sup
from receipts.check import check
from receipts.fetch import Fetcher
from receipts.parse import parse_report
from receipts.report import markdown, rows
from receipts.support import (Judge, SupportChecker, context, page_sentences, rank_sentences,
                              strip_attribution, windows)

PAGE = ("Call recording for Local Services. Recordings are kept for 30 days and can be played back by the "
        "account owner. In legal categories the retention period is 60 days, after which recordings are "
        "deleted automatically.\n\nBusinesses can turn off call recording in the settings page of their "
        "account at any time. Callers hear a short message before the call is connected.")


class WordChecker:
    """Fake checker: support = share of the claim's numbers and long words found in the premise."""
    name = "fake-checker"

    def score(self, pairs):
        out = []
        for premise, claim in pairs:
            terms = [w for w in sup._terms(claim) if len(w) > 3 or w.isdigit()]
            out.append(sum(t in sup._terms(premise) for t in terms) / max(1, len(terms)))
        return out


def fake_judge(answers):
    """A judge whose endpoint returns the given answers in turn (dicts, or raw strings)."""
    queue = list(answers)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        a = queue.pop(0) if queue else {"reason": "x", "evidence": [], "verdict": "not_supported"}
        content = a if isinstance(a, str) else json.dumps(a)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    j = Judge("http://judge.test/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    return j, seen


def test_page_sentences_and_retrieval():
    sents = page_sentences(PAGE)
    assert sents[1].startswith("Recordings are kept for 30 days")
    assert len(sents) == 5
    best = rank_sentences(sents, "legal categories retention 60 days")
    assert best[0] == 2
    assert windows(sents, [2], max_words=20) == [(2, 3)]
    assert windows(sents, [2], max_words=30) == [(1, 3)]
    assert context(sents, [2]) == [1, 2, 3]


def test_strip_attribution():
    assert strip_attribution("Google's help page says recordings are kept for 30 days.") == "Recordings are kept for 30 days."
    assert strip_attribution("According to the FTC, fake reviews are illegal.") == "Fake reviews are illegal."
    assert strip_attribution("Recordings are kept for 30 days.") == "Recordings are kept for 30 days."
    assert strip_attribution("Google says so.") == "Google says so."           # too little left: unchanged
    assert strip_attribution("North Carolina: the article says the lawyer must get consent.") == "The lawyer must get consent."
    assert strip_attribution("Under the 2025 LSA terms, Google may use recorded calls.") == "Google may use recorded calls."
    assert strip_attribution("Wisconsin ethics counsel (Inside Track, Jan 2021) wrote that a lawyer discloses "
                             "client information.") == "A lawyer discloses client information."


def test_checker_alone_uses_the_threshold_and_shows_evidence():
    s = SupportChecker(WordChecker())
    good = s.assess("Recordings in legal categories are kept for 60 days.", [], "u", PAGE)
    assert good.verdict == "supported" and any("60 days" in e for e in good.evidence)
    bad = s.assess("Google sells call recordings to insurance companies every quarter.", [], "u", PAGE)
    assert bad.verdict == "not_supported"
    assert s.assess("anything", [], "u", "").verdict == "not_checked"


def test_judge_answer_is_used_and_evidence_comes_from_the_page():
    j, seen = fake_judge([{"reason": "says 60 days", "evidence": [3, 99], "verdict": "supported"}])
    s = SupportChecker(None, j).assess("Legal recordings are kept for 60 days.", [], "u", PAGE)
    assert s.verdict == "supported" and s.judge_verdict == "supported"
    assert s.evidence == [page_sentences(PAGE)[2]]                      # 99 is not a sentence shown: dropped
    body = seen[0]
    assert body["response_format"]["type"] == "json_schema"
    assert "[3] In legal categories" in body["messages"][1]["content"]


def test_malformed_judge_answer_is_retried_then_falls_back():
    j, _ = fake_judge(["not json", {"reason": "r", "evidence": [], "verdict": "maybe"}])
    s = SupportChecker(WordChecker(), j).assess("Legal recordings are kept for 60 days.", [], "u", PAGE)
    assert s.judge_verdict == "" and "no usable answer" in s.flag
    assert s.verdict in ("supported", "not_supported")


def test_disagreement_is_flagged():
    j, _ = fake_judge([{"reason": "r", "evidence": [], "verdict": "not_supported"}])
    s = SupportChecker(WordChecker(), j).assess("Recordings in legal categories are kept for 60 days.", [], "u", PAGE)
    assert s.verdict == "not_supported" and "checker model says supported" in s.flag


def test_majority_vote_and_split_vote():
    sup_ = {"reason": "r", "evidence": [2], "verdict": "supported"}
    not_ = {"reason": "r", "evidence": [], "verdict": "not_supported"}
    part = {"reason": "r", "evidence": [], "verdict": "partly"}
    j, _ = fake_judge([sup_, not_, sup_])
    j.samples = 3
    a = j.judge("c", [(1, "a"), (2, "b")])
    assert a.verdict == "supported" and a.votes == ["supported", "not_supported", "supported"]
    j, _ = fake_judge([sup_, not_, part])
    j.samples = 3
    assert "split vote" in j.judge("c", [(1, "a"), (2, "b")]).reason


REPORT = """Legal-category recordings are kept for 60 days [1].
Google sells recordings to insurers every quarter [1].
The settings page lists every recorded call [2].

## Sources
1. https://example.org/article
2. https://example.org/forbidden
"""


def test_check_with_support_reports_unsupported_claims(fetcher):
    r = check(parse_report(REPORT), fetcher, support=SupportChecker(WordChecker()))
    verdicts = [c.support.verdict for c in r.claims]
    assert verdicts == ["supported", "not_supported", "not_checked"]
    assert "support: 1 supported, 1 not supported, 1 not checked" in r.summary()
    md = markdown(r)
    assert "## Claims the cited page doesn't support" in md and "sells recordings" in md
    assert "checker model fake-checker" in md
    assert {row["support"] for row in rows(r)} == {"supported", "not_supported", "not_checked"}


def test_cli_with_judge_only(tmp_path, web, monkeypatch):
    report = tmp_path / "r.md"
    report.write_text(REPORT)
    client = httpx.Client(transport=httpx.MockTransport(web), follow_redirects=True)
    real_fetcher, real_judge = cli.Fetcher, sup.Judge
    monkeypatch.setattr(cli, "Fetcher", lambda cache, **kw: real_fetcher(cache, client=client))
    j, _ = fake_judge([{"reason": "r", "evidence": [3], "verdict": "supported"},
                       {"reason": "r", "evidence": [], "verdict": "not_supported"}])
    monkeypatch.setattr(sup, "Judge", lambda *a, **kw: j)
    out = tmp_path / "out"
    code = cli.main(["check", str(report), "-o", str(out), "--cache", str(tmp_path / "c"),
                     "--checker", "none", "--judge-url", "http://judge.test/v1", "--fail-on", "not_supported"])
    assert code == 1
    data = json.loads((out / "check.json").read_text())
    assert [c["support"]["verdict"] for c in data["claims"]] == ["supported", "not_supported", "not_checked"]


def test_checker_none_without_judge_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["check", "x.md", "--support", "--checker", "none"])


def test_a_claim_citing_two_pages_is_judged_on_both():
    other = "Unrelated text about gardening and compost. " * 3 + "Callers can ask for a copy of the recording by email."
    j, seen = fake_judge([{"reason": "both", "evidence": [1, 4], "verdict": "supported"}])
    s = SupportChecker(None, j).assess_pages("Recordings are kept 30 days and callers can ask for a copy.", [],
                                             [("https://a.example/x", PAGE), ("https://b.example/y", other)])
    shown = seen[0]["messages"][1]["content"]
    assert "(from a.example)" in shown and "(from b.example)" in shown
    assert s.verdict == "supported" and len(s.evidence) == 2


# ---- splitting dense claims ----------------------------------------------------------------------

DENSE = ("Recordings are kept for 30 days and can be played back by the account owner, legal categories "
         "keep them for 60 days, and callers hear a message before the call, according to 3 help pages "
         "(arXiv 2609.05059).")


def test_figures_density_and_fact_validation():
    assert sup.numbers("50.7% of 4,000 firms (arXiv 2609.05059v2)") == {"50.7", "4000"}
    assert sup.is_dense(DENSE) and sup.is_dense("A | B") and not sup.is_dense("Recordings are kept 30 days.")
    assert sup.fact_from_claim("Legal categories keep recordings for 60 days.", DENSE)
    assert not sup.fact_from_claim("Legal categories keep recordings for 90 days.", DENSE)      # new figure
    assert not sup.fact_from_claim("Google sells recordings to insurers.", DENSE)              # new words


def test_a_dense_claim_is_judged_fact_by_fact():
    j, seen = fake_judge([
        {"facts": ["Recordings are kept for 30 days.", "Legal categories keep recordings for 60 days.",
                   "Google sells recordings to insurers for 5 dollars."]},          # invented: dropped
        {"reason": "says 30 days", "evidence": [2], "verdict": "supported"},
        {"reason": "no such sentence", "evidence": [], "verdict": "not_supported"},
    ])
    s = SupportChecker(None, j).assess(DENSE, [], "https://a.example/x", PAGE)
    assert [f["verdict"] for f in s.facts] == ["supported", "not_supported"]
    assert s.verdict == "partly" and s.reason.startswith("1 of 2 facts supported")
    assert s.left_out == ["3"] and "no checked fact covers: 3" in s.flag
    assert "split a claim" in seen[0]["messages"][0]["content"]


def test_all_facts_supported_is_supported_and_a_failed_split_judges_whole():
    j, _ = fake_judge([{"facts": ["Recordings are kept for 30 days.", "Legal categories keep recordings for 60 days.",
                                  "Callers hear a message before the call."]},
                       *[{"reason": "r", "evidence": [1], "verdict": "supported"}] * 3])
    s = SupportChecker(None, j).assess(DENSE, [], "u", PAGE)
    assert s.verdict == "supported" and len(s.facts) == 3
    j, _ = fake_judge(["no json", "still none", {"reason": "whole", "evidence": [2], "verdict": "partly"}])
    s = SupportChecker(None, j).assess(DENSE, [], "u", PAGE)
    assert s.facts == [] and s.verdict == "partly" and s.reason == "whole"


def test_split_is_off_without_a_judge_or_when_asked():
    assert SupportChecker(WordChecker()).split is False
    j, seen = fake_judge([{"reason": "whole", "evidence": [], "verdict": "partly"}])
    SupportChecker(None, j, split=False).assess(DENSE, [], "u", PAGE)
    assert len(seen) == 1 and "CLAIM" in seen[0]["messages"][1]["content"]


def test_a_checker_failure_is_flagged_not_fatal():
    class Broken:
        name = "broken"

        def score(self, pairs):
            raise RuntimeError("device lost")
    s = SupportChecker(Broken()).assess("Recordings are kept for 30 days.", [], "u", PAGE)
    assert s.verdict == "not_checked" and "checker model failed" in s.flag
    j, _ = fake_judge([{"reason": "r", "evidence": [2], "verdict": "supported"}])
    s = SupportChecker(Broken(), j).assess("Recordings are kept for 30 days.", [], "u", PAGE)
    assert s.verdict == "supported" and "checker model failed" in s.flag
