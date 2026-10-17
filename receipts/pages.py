"""Turn a cited URL into the text a reader of that page would see."""
from __future__ import annotations

import html
import io
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field

from .fetch import Fetched, Fetcher
from .text import html_to_text, loose, words

MIN_READABLE = 400          # letters and digits; less than this is a blocked or script-rendered page

ARCTIC = "https://arctic-shift.photon-reddit.com/api/"
_REDDIT = re.compile(r"reddit\.com/r/[^/]+/comments/([a-z0-9]+)", re.I)


@dataclass
class Page:
    url: str
    status: int
    text: str
    fetched: list[Fetched]
    via: str = "direct"                     # or "arctic-shift"
    error: str = ""
    _loose: str = field(default="", repr=False)
    _words: list[str] = field(default_factory=list, repr=False)
    _spans: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _base: str = field(default="", repr=False)
    _shingles: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._loose = loose(self.text)
        self._base = unicodedata.normalize("NFKC", html.unescape(self.text))
        ms = list(re.finditer(r"[A-Za-z0-9]+", self._base))
        self._spans = [m.span() for m in ms]
        self._words = [m.group().lower() for m in ms]
        for i in range(len(self._words) - 4):
            self._shingles.setdefault(" ".join(self._words[i:i + 5]), i)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def readable(self) -> bool:
        if self.via == "arctic-shift":              # archive text is real text, however short the thread
            return self.ok and bool(self._loose)
        return self.ok and len(self._loose) >= MIN_READABLE

    @property
    def sha256(self) -> str:
        return self.fetched[-1].sha256 if self.fetched else ""

    @property
    def fetched_at(self) -> str:
        return self.fetched[-1].fetched_at if self.fetched else ""

    def contains(self, quote: str) -> bool:
        q = loose(quote)
        return bool(q) and q in self._loose

    def passage(self, start: int, end: int) -> str:
        """Page text from word `start` to word `end` (exclusive), as written on the page."""
        a, b = self._spans[start][0], self._spans[end - 1][1]
        return re.sub(r"\s+", " ", self._base[a:b]).strip()


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    except Exception:
        pass
    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            return subprocess.run(["pdftotext", "-layout", tmp.name, "-"],
                                  capture_output=True, text=True).stdout
    return ""


def body_text(data: bytes, content_type: str) -> str:
    if data[:5] == b"%PDF-" or "pdf" in content_type.lower():
        return _pdf_text(data)
    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    try:
        t = data.decode(m.group(1) if m else "utf-8", "replace")
    except LookupError:
        t = data.decode("utf-8", "replace")
    if t.lstrip()[:1] in "{[":                  # JSON endpoints: keep every string value
        return t.replace("\\n", " ").replace('\\"', '"')
    if "html" in content_type.lower() or re.search(r"(?i)<(html|body|p|div)\b", t[:5000]):
        return html_to_text(t)
    return t


def _reddit(url: str, tid: str, fetcher: Fetcher) -> Page:
    """Reddit refuses plain requests, so a thread is read from the Arctic Shift archive instead."""
    parts, got = [], []
    for api in (f"{ARCTIC}posts/ids?ids={tid}",
                f"{ARCTIC}comments/search?link_id={tid}&limit=100"):
        f = fetcher.get(api)
        got.append(f)
        if f.status != 200:
            return Page(url, f.status, "", got, via="arctic-shift", error=f.error)
        try:
            for item in json.loads(f.body()).get("data", []):
                parts += [str(item.get(k, "")) for k in ("title", "selftext", "body")]
        except (ValueError, AttributeError):
            return Page(url, 0, "", got, via="arctic-shift", error="archive returned no JSON")
    return Page(url, 200, "\n".join(parts), got, via="arctic-shift")


def read_page(url: str, fetcher: Fetcher) -> Page:
    if not url.lower().startswith(("http://", "https://")):
        return Page(url, 0, "", [], error="not a web address")
    m = _REDDIT.search(url)
    if m:
        return _reddit(url, m.group(1).lower(), fetcher)
    f = fetcher.get(url)
    text = body_text(f.body(), f.content_type) if f.status else ""
    return Page(url, f.status, text, [f], error=f.error)
