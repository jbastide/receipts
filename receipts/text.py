"""Text normalisation shared by the quote grader and the report parser.

`norm`, `loose` and `words` are the same rules the published studies used, so a quote graded
here gets the grade it got there.
"""
import html
import re
import unicodedata

_SWAPS = (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
          ("–", "-"), ("—", "-"), ("‑", "-"), (" ", " "), ("​", ""))


def norm(s: str) -> str:
    """Case-fold, straighten quotes and dashes, collapse whitespace."""
    s = unicodedata.normalize("NFKC", html.unescape(s))
    for a, b in _SWAPS:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def loose(s: str) -> str:
    """Letters and digits only: ignores spacing, punctuation and case."""
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(s))


def html_to_text(raw: str) -> str:
    """Drop scripts, styles and tags. Crude on purpose: a quote is looked for in what a reader sees."""
    t = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", raw)
    t = re.sub(r"(?is)<!--.*?-->", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return html.unescape(t)


def squash(s: str, limit: int = 0) -> str:
    """One line, single spaces, optionally cut to `limit` characters."""
    s = re.sub(r"\s+", " ", s).strip()
    return s if not limit or len(s) <= limit else s[: limit - 1].rstrip() + "…"
