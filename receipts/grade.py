"""Grade a quote against the page it cites, and a link against the web and the Wayback Machine.

Quote grades
  verbatim    the whole quote is on the page (spacing, punctuation and case ignored)
  elided      the quote has an ellipsis and every piece of it is on the page
  edited      at least 80% of the quote's five-word runs are on the page: trimmed, reworded or stitched
  missing     the page was read and the quote is not there
  unreadable  the page could not be read: refused, gone, or too little text (script-rendered)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote as urlquote

from .fetch import Fetcher
from .pages import Page
from .text import loose, words

QUOTE_GRADES = ("verbatim", "elided", "edited", "missing", "unreadable")
EDITED_SHARE = 0.8
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\s*){3}\]|(?:\.\s*){3}|…)\s*")


@dataclass
class QuoteGrade:
    grade: str
    share: float = 0.0          # share of the quote's five-word runs found on the page
    closest: str = ""           # the page's nearest passage, for edited and missing quotes


def _runs(ws: list[str]) -> list[str]:
    return [" ".join(ws[i:i + 5]) for i in range(max(1, len(ws) - 4))]


def _closest(page: Page, hits: list[int], n_words: int) -> str:
    """The stretch of page holding the most matched runs, about as long as the quote."""
    if not hits:
        return ""
    hits.sort()
    span = n_words + 5
    best, best_i, best_j = 0, 0, 0
    j = 0
    for i, h in enumerate(hits):
        while hits[j] < h - span:
            j += 1
        if i - j + 1 > best:
            best, best_i, best_j = i - j + 1, i, j
    start = hits[best_j]
    end = max(hits[best_i] + 5, start + n_words)       # run on past the last match, where the edit is
    return page.passage(start, min(end, len(page._words)))


def grade_quote(quote: str, page: Page) -> QuoteGrade:
    if page.ok and page.contains(quote):
        return QuoteGrade("verbatim", 1.0)
    if not page.readable:
        return QuoteGrade("unreadable")
    pieces = [p for p in _ELLIPSIS.split(quote) if loose(p)]
    if len(pieces) > 1 and all(page.contains(p) for p in pieces):
        return QuoteGrade("elided", 1.0)
    ws = words(quote)
    runs = _runs(ws)
    if len(ws) < 5:                                    # too short for runs: whole-phrase match only
        return QuoteGrade("missing", 0.0)
    hits = [page._shingles[r] for r in runs if r in page._shingles]
    share = round(len(hits) / len(runs), 2)
    grade = "edited" if share >= EDITED_SHARE else "missing"
    return QuoteGrade(grade, share, _closest(page, hits, len(ws)))


# ---- links ---------------------------------------------------------------------------------

LINK_STATES = ("ok", "unreadable", "refused", "gone", "error")
_REFUSED = {401, 402, 403, 407, 418, 429, 451, 999}


def link_state(page: Page) -> str:
    """ok, unreadable (thin or script-rendered), refused (blocked us), gone (404/410), error."""
    if page.readable:
        return "ok"
    if page.ok:
        return "unreadable"
    if page.status in _REFUSED:
        return "refused"
    if page.status in (404, 410):
        return "gone"
    return "error"


@dataclass
class Archive:
    state: str                  # "archived", "none" or "unknown" (the lookup itself failed)
    first_capture: str = ""     # YYYYMMDDhhmmss of the earliest Wayback capture
    snapshot: str = ""


def wayback(url: str, fetcher: Fetcher) -> Archive:
    """Has the Wayback Machine ever captured this URL? Tells a page that moved from one that never was."""
    api = ("https://web.archive.org/cdx/search/cdx?output=json&limit=1&fl=timestamp,original&url="
           + urlquote(url.split("#")[0], safe=""))
    f = fetcher.get(api)
    if f.status != 200:
        return Archive("unknown")
    try:
        rows = json.loads(f.body() or b"[]")
    except ValueError:
        return Archive("unknown")
    if len(rows) < 2:
        return Archive("none")
    ts, original = rows[1][0], rows[1][1]
    return Archive("archived", ts, f"https://web.archive.org/web/{ts}/{original}")
