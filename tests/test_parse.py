import json

import pytest

from receipts.parse import parse_report


def by_text(claims, needle):
    return next(c for c in claims if needle in c.text)


def test_inline_link_and_quote_in_one_sentence():
    md = ('Google says calls “may be monitored or recorded to detect spam” '
          '([Google](https://example.org/article)). Nothing else here.')
    [c] = parse_report(md)
    assert c.urls == ["https://example.org/article"]
    assert c.quotes == ["may be monitored or recorded to detect spam"]
    assert "https://" not in c.text


def test_numbered_markers_resolve_through_a_sources_list():
    md = """Recordings are kept for 30 days [1]. Legal categories keep them for 60 days [2][3].

Callers hear a message.[1] Businesses can turn it off [1, 3].

## Sources
1. Google help: https://example.org/article
2. [Terms](https://example.org/terms)
3. https://example.org/other
"""
    cs = parse_report(md)
    assert by_text(cs, "30 days").urls == ["https://example.org/article"]
    assert by_text(cs, "60 days").urls == ["https://example.org/terms", "https://example.org/other"]
    assert by_text(cs, "Callers hear").urls == ["https://example.org/article"]
    assert by_text(cs, "turn it off").urls == ["https://example.org/article", "https://example.org/other"]
    assert not any("Sources" in c.text for c in cs)


def test_footnotes_and_reference_definitions():
    md = """A claim with a footnote.[^a] Another with a reference link [the page][ref].

[^a]: See https://example.org/article for details.
[ref]: https://example.org/other
"""
    cs = parse_report(md)
    assert by_text(cs, "footnote").urls == ["https://example.org/article"]
    assert by_text(cs, "reference link").urls == ["https://example.org/other"]


def test_quote_without_its_own_link_borrows_the_next_one_and_says_so():
    md = ('The page says “recordings are kept for 30 days and can be played back”. '
          'That comes from Google’s help center (https://example.org/article).')
    cs = parse_report(md)
    q = next(c for c in cs if c.quotes)
    assert q.urls == ["https://example.org/article"]
    assert "same paragraph" in q.note


def test_short_quoted_terms_are_not_quotes():
    [c] = parse_report('This is what Google calls “AI Mode” in its docs (https://example.org/article).')
    assert c.quotes == []


def test_block_quote_takes_the_link_from_the_paragraph_before():
    md = """Google's help page (https://example.org/article) puts it this way:

> Businesses can turn off call recording in the settings page of their account at any time.
"""
    cs = parse_report(md)
    bq = next(c for c in cs if c.quotes)
    assert bq.urls == ["https://example.org/article"]
    assert bq.quotes[0].startswith("Businesses can turn off")


def test_list_items_and_table_rows_are_separate_claims():
    md = """- First finding (https://example.org/a)
- Second finding (https://example.org/b)

| Fact | Source |
| --- | --- |
| Thirty days | https://example.org/c |
"""
    cs = parse_report(md)
    assert [c.urls[0] for c in cs] == ["https://example.org/a", "https://example.org/b", "https://example.org/c"]


def test_abbreviations_do_not_split_sentences_and_code_is_ignored():
    md = """Agencies, e.g. the big ones, record calls “for quality and training purposes only” (https://example.org/a).

```
not a claim https://example.org/code
```
"""
    [c] = parse_report(md)
    assert c.quotes == ["for quality and training purposes only"]


def test_receipts_json_and_code_fenced_json():
    data = {"claims": [{"id": "c1", "claim": "x", "url": "https://example.org/a", "quote": "a b c d"}]}
    for raw in (json.dumps(data), "```json\n" + json.dumps(data) + "\n```", json.dumps(data["claims"])):
        [c] = parse_report(raw)
        assert (c.id, c.urls, c.quotes) == ("c1", ["https://example.org/a"], ["a b c d"])


def test_openai_responses_annotations():
    text = "Calls may be recorded to detect spam. Recordings are kept for 30 days."
    data = {"output": [{"type": "message", "content": [{"type": "output_text", "text": text, "annotations": [
        {"type": "url_citation", "start_index": 0, "end_index": 37, "url": "https://example.org/a", "title": "A"},
        {"type": "url_citation", "start_index": 38, "end_index": len(text), "url": "https://example.org/b", "title": "B"}]}]}]}
    cs = parse_report(json.dumps(data))
    assert by_text(cs, "spam").urls == ["https://example.org/a"]
    assert by_text(cs, "30 days").urls == ["https://example.org/b"]


def test_anthropic_cited_text_becomes_the_quote():
    data = {"type": "message", "content": [
        {"type": "text", "text": "Google may record calls."},
        {"type": "text", "text": "Calls may be recorded to detect spam.", "citations": [
            {"type": "web_search_result_location", "url": "https://example.org/a", "title": "A",
             "cited_text": "may be monitored or recorded to detect spam"}]}]}
    [c] = parse_report(json.dumps(data))
    assert c.urls == ["https://example.org/a"]
    assert c.quotes == ["may be monitored or recorded to detect spam"]


def test_perplexity_citations_list():
    data = {"choices": [{"message": {"content": "Calls may be recorded [1]. Kept 30 days [2]."}}],
            "citations": ["https://example.org/a", "https://example.org/b"]}
    cs = parse_report(json.dumps(data))
    assert [c.urls for c in cs] == [["https://example.org/a"], ["https://example.org/b"]]


def test_unknown_json_is_an_error():
    with pytest.raises(ValueError):
        parse_report(json.dumps({"something": "else"}))


def test_clauses_with_their_own_sources_become_separate_claims():
    md = ('Reasoning models were no less biased ([A](https://example.org/a)); large models '
          '"show no improvement in robustness to bandwagon cues" ([B](https://example.org/b)); and a '
          '"focus only on product features" prompt left effects intact ([C](https://example.org/c)).')
    cs = parse_report(md)
    assert [c.urls for c in cs] == [["https://example.org/a"], ["https://example.org/b"], ["https://example.org/c"]]
    assert cs[1].quotes == ["show no improvement in robustness to bandwagon cues"]


def test_a_quote_is_tried_against_the_link_after_it_first():
    md = ('Earlier work ([A](https://example.org/a)) found that agents "act as switches that override '
          'other cues" ([B](https://example.org/b)) in most runs.')
    [c] = parse_report(md)
    assert c.quote_urls == [["https://example.org/b", "https://example.org/a"]]
