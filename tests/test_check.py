import csv
import hashlib
import json

import httpx
import pytest

from receipts import cli
from receipts.check import check
from receipts.fetch import Fetcher
from receipts.parse import parse_report
from receipts.report import markdown

REPORT = """# Does Google record calls?

Google says calls “may be monitored or recorded to detect spam or unauthorized behavior” [1].
It also says “in legal categories the retention period is 60 days, after which recordings are deleted manually” [1].
Google adds that “recordings are shared with the police on request” [1].
A bar committee found that recorded calls “raise questions of confidentiality for prospective clients” [2].
One firm owner wrote that “you can switch it off in the call settings” [3].
A trade blog claims “every call is transcribed by default” [4].
An older guide covered the same ground [5], and a vendor post did too [6].
The settings help page is built in JavaScript [7].

## Sources
1. https://example.org/article
2. https://example.org/opinion.pdf
3. https://www.reddit.com/r/LawFirm/comments/abc123/does_google_record/
4. https://example.org/forbidden
5. https://example.org/moved
6. https://example.org/never
7. https://example.org/script-app
"""


@pytest.fixture
def result(fetcher):
    return check(parse_report(REPORT), fetcher, source="report.md")


def grades(result):
    return {c.claim.id: [q.grade for q in c.quotes] for c in result.claims}


def test_every_kind_of_result(result):
    assert grades(result) == {"c1": ["verbatim"], "c2": ["edited"], "c3": ["missing"], "c4": ["verbatim"],
                              "c5": ["verbatim"], "c6": ["unreadable"], "c7": [], "c8": []}
    states = {u.rsplit("/", 1)[-1]: l.state for u, l in result.links.items()}
    assert states["forbidden"] == "refused" and states["moved"] == "gone" and states["script-app"] == "unreadable"
    assert result.links["https://example.org/moved"].archive.state == "archived"
    assert result.links["https://example.org/never"].archive.state == "none"
    assert result.links["https://example.org/article"].archive is None


def test_summary_line(result):
    s = result.summary()
    assert s.startswith("8 claims citing 7 links")
    assert "quotes: 6 (3 word for word, 1 edited, 1 missing, 1 unreadable)" in s
    assert "1 with no Wayback record" in s


def test_markdown_report_explains_problems(result):
    md = markdown(result, "/tmp/cache")
    assert "## Quotes to look at" in md
    assert "after which recordings are deleted automatically" in md   # nearest passage for the edited quote
    assert "the page may never have existed" in md
    assert "first captured 2021-03-04" in md
    assert "Arctic Shift" in md
    assert "does not judge whether a page supports a claim" in md


def test_pages_are_cached_with_hash_and_reused_offline(tmp_path, web, make_fetcher):
    claims = parse_report(REPORT)
    first = check(claims, make_fetcher())
    n = len(web.calls)
    second = check(claims, make_fetcher(offline=True))
    assert len(web.calls) == n                                       # nothing fetched the second time
    assert grades(second) == grades(first)
    meta = next((tmp_path / "cache").glob("*.json"))
    m = json.loads(meta.read_text())
    body = (tmp_path / "cache" / meta.name.replace(".json", ".body")).read_bytes()
    assert m["sha256"] == hashlib.sha256(body).hexdigest() and m["fetched_at"].endswith("+00:00")


def test_offline_without_cache_is_unreadable_not_missing(tmp_path):
    f = Fetcher(str(tmp_path / "empty"), offline=True)
    r = check(parse_report(REPORT), f, use_wayback=False)
    assert {g for gs in grades(r).values() for g in gs} == {"unreadable"}


def test_cli_writes_three_files_and_fail_on(tmp_path, web, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text(REPORT)
    client = httpx.Client(transport=httpx.MockTransport(web), follow_redirects=True)
    real = Fetcher

    def fake_fetcher(cache, **kw):
        return real(cache, client=client, offline=kw.get("offline", False))

    monkeypatch.setattr(cli, "Fetcher", fake_fetcher)
    out = tmp_path / "out"
    code = cli.main(["check", str(report), "-o", str(out), "--cache", str(tmp_path / "c")])
    assert code == 0
    rows = list(csv.DictReader(open(out / "check.csv")))
    assert {r["quote_grade"] for r in rows} == {"verbatim", "edited", "missing", "unreadable", ""}
    assert json.loads((out / "check.json").read_text())["quote_counts"]["edited"] == 1
    assert cli.main(["check", str(report), "-o", str(out), "--cache", str(tmp_path / "c"),
                     "--fail-on", "edited"]) == 1
    assert cli.main(["check", str(report), "-o", str(out), "--cache", str(tmp_path / "c"),
                     "--fail-on", "elided"]) == 0
