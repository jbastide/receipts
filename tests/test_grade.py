from receipts.grade import grade_quote, link_state, wayback
from receipts.pages import read_page
from receipts.text import loose, norm, words

QUOTE = ("All phone calls, emails, and text-based communications delivered through Google may be "
         "monitored or recorded to detect spam or unauthorized behavior")


def test_normalisation_ignores_case_punctuation_and_curly_quotes():
    assert norm("“Hello”—World’s  test") == "\"hello\"-world's test"
    assert loose("Text-based, communications!") == loose("text based communications")
    assert words("It’s 30 days.") == ["it", "s", "30", "days"]


def test_verbatim_ignores_spacing_and_punctuation(fetcher):
    page = read_page("https://example.org/article", fetcher)
    assert grade_quote(QUOTE, page).grade == "verbatim"
    assert grade_quote(QUOTE.upper().replace(",", ""), page).grade == "verbatim"


def test_script_text_is_not_page_text(fetcher):
    page = read_page("https://example.org/article", fetcher)
    assert grade_quote("var x = not page text here", page).grade == "missing"


def test_one_word_changed_is_edited_and_nearest_passage_shown(fetcher):
    page = read_page("https://example.org/article", fetcher)
    q = "In legal categories the retention period is 60 days, after which recordings are deleted automatically"
    g = grade_quote(q.replace("automatically", "manually"), page)
    assert g.grade == "edited"
    assert 0.8 <= g.share < 1
    assert "after which recordings are deleted" in g.closest
    g = grade_quote(QUOTE.replace("spam", "fraud"), page)               # a change mid-quote breaks five runs
    assert g.grade == "missing" and 0.5 < g.share < 0.8 and "detect spam" in g.closest


def test_ellipsis_with_every_piece_on_the_page_is_elided(fetcher):
    page = read_page("https://example.org/article", fetcher)
    q = "Recordings are kept for 30 days … the retention period is 60 days"
    assert grade_quote(q, page).grade == "elided"
    assert grade_quote("Recordings are kept for 30 days ... the retention period is 90 days", page).grade != "elided"


def test_stitched_passages_are_not_verbatim(fetcher):
    page = read_page("https://example.org/article", fetcher)
    q = "Recordings are kept for 30 days and callers hear a short message before the call is connected"
    assert grade_quote(q, page).grade in ("edited", "missing")


def test_quote_from_another_page_is_missing(fetcher):
    page = read_page("https://example.org/other", fetcher)
    g = grade_quote(QUOTE, page)
    assert g.grade == "missing" and g.share == 0 and g.closest == ""


def test_negative_control_one_word_replaced_is_never_verbatim(fetcher):
    page = read_page("https://example.org/article", fetcher)
    ws = QUOTE.split()
    for i in range(1, len(ws) - 1):
        damaged = " ".join(ws[:i] + ["zebra"] + ws[i + 1:])
        assert grade_quote(damaged, page).grade != "verbatim"


def test_thin_refused_and_dead_pages_are_unreadable(fetcher):
    for url, state in (("https://example.org/script-app", "unreadable"),
                       ("https://example.org/forbidden", "refused"),
                       ("https://example.org/never", "gone"),
                       ("https://nowhere.invalid/x", "error")):
        page = read_page(url, fetcher)
        assert link_state(page) == state
        assert grade_quote(QUOTE, page).grade == "unreadable"


def test_wayback_separates_moved_from_never_captured(fetcher):
    moved = wayback("https://example.org/moved", fetcher)
    assert moved.state == "archived" and moved.first_capture.startswith("2021")
    assert wayback("https://example.org/never", fetcher).state == "none"


def test_pdf_text_is_read(fetcher):
    page = read_page("https://example.org/opinion.pdf", fetcher)
    assert page.readable
    q = "recorded calls from advertising platforms raise questions of confidentiality for prospective clients"
    assert grade_quote(q, page).grade == "verbatim"


def test_reddit_threads_are_read_from_the_archive(fetcher, web):
    page = read_page("https://www.reddit.com/r/LawFirm/comments/AbC123/does_google_record/", fetcher)
    assert page.via == "arctic-shift"
    assert grade_quote("you can switch it off in the call settings", page).grade == "verbatim"
    assert not any("www.reddit.com" in c for c in web.calls)


def test_google_help_pages_are_requested_in_english(fetcher, web):
    page = read_page("https://support.google.com/localservices/answer/7281751", fetcher)
    assert grade_quote(QUOTE, page).grade == "verbatim"
    assert any("hl=en" in c for c in web.calls)
