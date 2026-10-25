import csv
import json

import httpx
import pytest

from receipts import cli
from receipts import study as st
from receipts.collect import reddit_threads
from receipts.fetch import Fetcher

TOML = '''name = "test study"
unit_label = "thread"
instructions = "Code the thread."
items = "items.jsonl"
gold = "gold.csv"
calibration_ids = "cal.txt"
min_kappa = 0.6
chunk_words = 12

[[fields]]
name = "PRICE"
definition = "Someone complains about price."

[[fields]]
name = "BUGS"
definition = "Someone reports a bug."

[coders.a]
url = "http://coder.test/v1"
model = "a"

[coders.b]
url = "http://coder.test/v1"
model = "b"
'''

ITEMS = [
    {"id": "t1", "meta": {"title": "Pricing"}, "units": [
        {"id": "post", "text": "Is this tool worth it"},
        {"id": "c:1", "text": "way too expensive for what it does honestly"},
        {"id": "c:2", "text": "it crashed twice on export yesterday"}]},
    {"id": "t2", "meta": {"title": "Hello"}, "units": [{"id": "post", "text": "Just saying hello"}]},
    {"id": "t3", "meta": {"title": "Export"}, "units": [
        {"id": "post", "text": "Export is broken"}, {"id": "c:9", "text": "same bug here"}]},
]


def keyword_coder(name, wrong_unit=False):
    """A fake model: codes PRICE for 'expensive' and BUGS for 'crash', 'broken' or 'bug'."""
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        user = body["messages"][1]["content"]
        codes = []
        for line in user.splitlines():
            if not line.startswith("["):
                continue
            n = int(line[1:line.index("]")])
            if "expensive" in line:
                codes.append({"field": "PRICE", "unit": 99 if wrong_unit else n})
            if any(w in line for w in ("crash", "broken", "bug")) and name == "a":
                codes.append({"field": "BUGS", "unit": n})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"codes": codes})}}]})

    return st.Coder(name, "http://coder.test/v1", name, client=httpx.Client(transport=httpx.MockTransport(handler))), calls


@pytest.fixture
def study(tmp_path):
    (tmp_path / "study.toml").write_text(TOML)
    st.write_items(ITEMS, str(tmp_path / "items.jsonl"))
    with open(tmp_path / "gold.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "PRICE", "BUGS"])
        w.writerows([["t1", "true", "true"], ["t2", "false", "false"], ["t3", "false", "true"]])
    (tmp_path / "cal.txt").write_text("t1\nt2\n")
    return st.load_study(str(tmp_path / "study.toml"))


def test_chunks_keep_the_post_and_whole_units():
    units = [{"id": "post", "text": "a b c"}] + [{"id": f"c:{i}", "text": "w " * 5} for i in range(4)]
    assert st.chunks(units, 12) == [[0, 1], [0, 2], [0, 3], [0, 4]]
    assert st.chunks(units, 100) == [[0, 1, 2, 3, 4]]
    assert st.chunks([], 10) == []


def test_kappa():
    assert st.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert st.kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert st.kappa([False, False], [False, False]) is None


def test_coding_cites_units_and_or_across_chunks(study):
    coder, calls = keyword_coder("a")
    row = st.code_item(study, coder, ITEMS[0])
    assert row["chunks"] == 2 and len(calls) == 2                 # chunk_words = 12 splits t1
    assert row["values"] == {"PRICE": True, "BUGS": True}
    assert row["evidence"] == {"PRICE": ["c:1"], "BUGS": ["c:2"]}
    sys_prompt = calls[0]["messages"][0]["content"]
    assert "PRICE: Someone complains about price." in sys_prompt
    assert calls[0]["response_format"]["json_schema"]["schema"]["properties"]["codes"]["items"]["properties"]["field"]["enum"] == ["PRICE", "BUGS"]


def test_a_code_citing_a_unit_not_shown_is_dropped(study):
    coder, _ = keyword_coder("a", wrong_unit=True)
    row = st.code_item(study, coder, ITEMS[0])
    assert row["values"]["PRICE"] is False and row["dropped_codes"] == 1


def test_code_calibrate_compare_end_to_end(study):
    for name in ("a", "b"):
        coder, _ = keyword_coder(name)
        st.code_study(study, coder)
        st.code_study(study, coder)                                   # resumable: nothing coded twice
        assert len(st.read_codes(f"{study.out}/codes-{name}.jsonl")) == 3
    cal, held = st.calibrate(study, "a")
    assert [r["field"] for r in cal] == ["PRICE", "BUGS"] and cal[1]["passes"] is True
    assert held[0]["n"] == 1                                          # t3 is held out
    cal_b, _ = st.calibrate(study, "b")
    assert cal_b[1]["passes"] is False                                # b never codes BUGS
    rows = st.compare(study, "a", "b")
    assert {r["field"]: r["agree_pct"] for r in rows} == {"PRICE": 100.0, "BUGS": 33.3}
    review = list(csv.DictReader(open(f"{study.out}/review-a-b.csv")))
    assert {(r["id"], r["field"]) for r in review} == {("t1", "BUGS"), ("t3", "BUGS")}
    assert "crashed twice" in next(r for r in review if r["id"] == "t1")["text"]


def test_reddit_collector_pages_comments(tmp_path):
    def handler(request):
        url = str(request.url)
        if "posts/ids" in url:
            return httpx.Response(200, json={"data": [{"id": "abc", "subreddit": "SEO", "created_utc": 1735689600,
                                                       "title": "T", "selftext": "Body", "num_comments": 101}]})
        after = int(request.url.params["after"])
        if after < 2000000000:
            page = [{"id": f"k{i}", "body": f"comment {i}", "created_utc": 1735689600 + i} for i in range(100)]
            page[3]["body"] = "[deleted]"
            if after > 1735689600:                                     # second page: one new comment
                page = page[-1:] + [{"id": "last", "body": "the end", "created_utc": 1735689999}]
            return httpx.Response(200, json={"data": page})
        return httpx.Response(200, json={"data": []})

    f = Fetcher(str(tmp_path / "c"), client=httpx.Client(transport=httpx.MockTransport(handler)))
    [item] = reddit_threads(["abc", "missing"], f, pause=0)
    ids = [u["id"] for u in item["units"]]
    assert ids[0] == "post" and "c:k3" not in ids and ids[-1] == "c:last" and len(ids) == 1 + 99 + 1
    assert item["meta"]["date"] == "2025-01-01"


def test_cli_study_code_and_calibrate(study, monkeypatch, capsys):
    coder, _ = keyword_coder("a")
    monkeypatch.setattr(st, "Coder", lambda *a, **k: coder)
    assert cli.main(["study", "code", study.path, "--coder", "a"]) == 0
    assert cli.main(["study", "calibrate", study.path, "--coders", "a"]) == 0
    out = capsys.readouterr().out
    assert "3 items coded by a" in out and "BUGS" in out


def test_collector_retries_a_busy_archive_and_does_not_trust_a_stored_failure(tmp_path):
    answers = iter([httpx.Response(422, json={"data": None, "error": "Timeout. Maybe slow down a bit"}),
                    httpx.Response(200, json={"data": [{"id": "p", "subreddit": "s", "created_utc": 1735689600,
                                                        "title": "T", "selftext": "", "num_comments": 0}]}),
                    httpx.Response(200, json={"data": []})])
    f = Fetcher(str(tmp_path / "c"), client=httpx.Client(transport=httpx.MockTransport(lambda r: next(answers))))
    [item] = reddit_threads(["p"], f, pause=0)
    assert item["units"][0]["id"] == "post"


def test_gate_clears_fields_when_the_gate_does_not_apply(study):
    study.gate, study.ungated = "PRICE", []
    coder, _ = keyword_coder("a")
    row = st.code_item(study, coder, ITEMS[2])               # t3: BUGS only, no PRICE
    assert row["values"] == {"PRICE": False, "BUGS": False} and row["evidence"]["BUGS"] == []
    study.ungated = ["BUGS"]
    assert st.code_item(study, coder, ITEMS[2])["values"]["BUGS"] is True


def test_verify_rejects_proposed_codes_and_reapplies_the_gate(study):
    study.verify_instructions = "Check the code."
    coder, _ = keyword_coder("a")
    row = st.code_item(study, coder, ITEMS[0])                       # PRICE and BUGS proposed
    asked = []

    def handler(request):
        body = json.loads(request.content)
        asked.append(body["messages"][0]["content"])
        ok = "FIELD PRICE" in body["messages"][0]["content"] and "expensive" in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"applies": ok})}}]})

    v = st.Coder("v", "http://coder.test/v1", "v", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = st.verify_item(study, v, ITEMS[0], row)
    assert out["proposed"] == {"PRICE": True, "BUGS": True}
    assert out["values"] == {"PRICE": True, "BUGS": False} and out["verify_rejected"] == 1 and len(asked) == 2
    study.gate = "BUGS"
    assert st.verify_item(study, v, ITEMS[0], row)["values"] == {"PRICE": False, "BUGS": False}


def test_kappa_interval_is_wide_with_few_items_and_narrow_with_many():
    few = st.kappa_ci([True, False] * 5, [True, False] * 4 + [False, True])
    many = st.kappa_ci([True, False] * 200, [True, False] * 190 + [False, True] * 10)
    assert few[0] < 0.4 < few[1]
    assert many[1] - many[0] < few[1] - few[0] and many[0] > 0.8
