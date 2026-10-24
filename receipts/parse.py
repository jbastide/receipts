"""Read a research report and pull out its claims, the links each one cites, and any quotes.

Formats
  Markdown or plain text   what people paste from ChatGPT, Claude, Gemini, Perplexity or an
                           open-source research agent: inline links, bare URLs, [1]-style and
                           [^1]-style references, a numbered Sources list, block quotes
  Receipts JSON            {"claims": [{"id", "claim", "url", "quote"}]}, the shape our research
                           briefs ask for (a bare list of such objects also works)
  API responses            OpenAI Responses (url_citation annotations), Anthropic Messages
                           (citations with cited_text), Perplexity chat completions (citations list)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MIN_CLAIM_WORDS = 3         # a shorter "sentence" is a fragment left by splitting
MIN_QUOTE_WORDS = 5         # shorter quoted phrases are usually terms or scare quotes, not quotations


@dataclass
class Claim:
    id: str
    text: str
    urls: list[str]
    quotes: list[str] = field(default_factory=list)
    quote_urls: list[list[str]] = field(default_factory=list)   # per quote: links to try, nearest first
    line: int = 0
    note: str = ""


# ---- shared patterns -------------------------------------------------------------------------

_URL = re.compile(r"https?://[^\s<>\"'”’)\]]+")
_INLINE = re.compile(r"(?<!!)\[([^\]]*)\]\((https?://[^\s)]+)(?:\s+\"[^\"]*\")?\)")
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
_MARKER = re.compile(r"\[(\^?[\w.-]+(?:\s*[,–-]\s*\^?[\w.-]+)*)\](?![(:])")
_QUOTED = re.compile(r"“([^”]+)”|\"([^\"\n]+)\"")
_REFDEF = re.compile(r"^\s{0,3}\[(\^?[^\]]+)\]:\s*(.*)$")
_LISTREF = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+(.*)$")
_SOURCES_HEAD = re.compile(r"^\s*(?:#+\s*|\*\*)?\s*(sources|references|citations|works cited|"
                           r"bibliography|notes|links)\b[\s:*]*$", re.I)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_ABBREV = {"e.g", "i.e", "u.s", "u.k", "mr", "mrs", "ms", "dr", "no", "vs", "etc", "inc", "st",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
           "fig", "al", "approx", "ca"}


def _clean_url(u: str) -> str:
    return u.rstrip(".,;:!?*_")


def _first_url(s: str) -> str:
    m = _INLINE.search(s) or _AUTOLINK.search(s)
    if m:
        return _clean_url(m.group(m.lastindex))
    m = _URL.search(s)
    return _clean_url(m.group()) if m else ""


def _expand_marker(body: str) -> list[str]:
    """'1, 3-5' -> ['1', '3', '4', '5']; '^a' -> ['^a']."""
    out = []
    for part in re.split(r"\s*,\s*", body):
        m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if m and int(m.group(2)) - int(m.group(1)) < 50:
            out += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        else:
            out.append(part.strip())
    return out


# ---- markdown and plain text -------------------------------------------------------------------

def _references(lines: list[str]) -> tuple[dict[str, str], set[int]]:
    """Reference definitions and the numbered Sources list. Returns (label -> url, lines to skip)."""
    refs, skip = {}, set()
    in_sources = False
    for i, line in enumerate(lines):
        m = _REFDEF.match(line)
        if m:
            url = _first_url(m.group(2)) or _first_url(m.group(2).strip("<>"))
            if url:
                refs[m.group(1).strip()] = url
                refs.setdefault(m.group(1).strip().lower(), url)
            skip.add(i)
            continue
        if _SOURCES_HEAD.match(line):
            in_sources = True
            skip.add(i)
            continue
        if in_sources:
            if line.lstrip().startswith("#"):
                in_sources = False
                continue
            m = _LISTREF.match(line)
            if m:
                url = _first_url(m.group(3))
                if url:
                    refs.setdefault(m.group(1) or m.group(2), url)
            skip.add(i)
    return refs, skip


def _segments(lines: list[str], skip: set[int]) -> list[tuple[int, str, bool]]:
    """Paragraphs, list items, table rows and block quotes: (first line number, text, is_blockquote)."""
    segs, buf, start, bq = [], [], 0, False
    fence = False
    header: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            segs.append((start, " ".join(buf), bq))
        buf = []

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            fence = not fence
            flush()
            continue
        if fence or i in skip:
            flush()
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            flush()
            continue
        is_bq = line.lstrip().startswith(">")
        text = re.sub(r"^\s*>\s?", "", line) if is_bq else line
        if line.lstrip().startswith("|"):
            flush()
            if re.fullmatch(r"[\s|:-]+", line):
                continue
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if re.fullmatch(r"\s*\|?[\s|:-]+\|?\s*", nxt) and "-" in nxt:
                header = [re.sub(r"[*_`]", "", c) for c in cells]      # the row above the |---| line
                continue
            if header and len(header) == len(cells):
                cells = [f"{h}: {c}" if h else c for h, c in zip(header, cells) if c]
            segs.append((i + 1, " | ".join(c for c in cells if c), False))
            continue
        header = header if line.lstrip().startswith("|") else []
        new_item = bool(_LIST_ITEM.match(text))
        if buf and (is_bq != bq or new_item):
            flush()
        if not buf:
            start, bq = i + 1, is_bq
        buf.append(text.strip())
    flush()
    return segs


def _depth(s: str) -> int:
    """How many brackets and parentheses are open at the end of `s`."""
    return max(0, s.count("(") - s.count(")")) + max(0, s.count("[") - s.count("]"))


def _sentences(seg: str) -> list[str]:
    """Split after . ! or ? (and any citation markers that follow) when a new sentence starts,
    never inside parentheses or brackets ("(Smith et al. 2024)", "[Hou et al., ECIR 2024](…)")."""
    out, last = [], 0
    for m in re.finditer(r"[.!?][\"”’)]*((?:\s*\[[^\]\s]{1,12}\])*)(?:\s+(?=[A-Z0-9\"“‘(\[*_])|\s*$)", seg):
        word = re.search(r"([\w.]+)[.!?][\"”’)]*$", seg[last:m.start() + 1])
        if word and (word.group(1).lower().rstrip(".") in _ABBREV or len(word.group(1)) == 1):
            continue
        if _depth(seg[:m.start()]):
            continue
        cut = m.end(1) if m.group(1) else m.start() + 1
        out.append(seg[last:cut].strip())
        last = cut
    out.append(seg[last:].strip())
    return [s for s in out if s]


def _blank(m: re.Match) -> str:
    return " " * len(m.group())


def _cites_at(s: str, refs: dict[str, str]) -> list[tuple[int, str]]:
    """Every citation in `s` with its position, so a quote can be matched to the link nearest it."""
    def ref(label):
        return refs.get(label) or refs.get(label.lower())

    found = [(m.start(), _clean_url(m.group(2))) for m in _INLINE.finditer(s)]
    found += [(m.start(), _clean_url(m.group(1))) for m in _AUTOLINK.finditer(s)]
    stripped = _AUTOLINK.sub(_blank, _INLINE.sub(_blank, s))      # same length, so positions hold
    found += [(m.start(), _clean_url(m.group())) for m in _URL.finditer(stripped)]
    for m in re.finditer(r"\[([^\]]+)\]\[([^\]]*)\]", s):          # [text][label] and [label][]
        if m.group(2) and _MARKER.fullmatch(f"[{m.group(1)}]"):
            continue                                                  # [2][3]: two markers, handled below
        url = ref((m.group(2) or m.group(1)).strip())
        if url:
            found.append((m.start(), url))
    for m in _MARKER.finditer(_URL.sub(_blank, stripped)):
        found += [(m.start(), ref(label)) for label in _expand_marker(m.group(1)) if ref(label)]
    return sorted(found, key=lambda pu: pu[0])


def _unique(urls) -> list[str]:
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _citations(s: str, refs: dict[str, str]) -> list[str]:
    return _unique(u for _, u in _cites_at(s, refs))


def _quotes(s: str, refs: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Quotations in `s`, each with the sentence's links ordered nearest-first (after it, then before)."""
    cites = _cites_at(s, refs)
    out = []
    for m in _QUOTED.finditer(_URL.sub(_blank, s)):
        q = _INLINE.sub(lambda x: x.group(1), m.group(1) or m.group(2)).strip()
        if len(q.split()) < MIN_QUOTE_WORDS:
            continue
        after = [u for p, u in cites if p >= m.end()]
        before = [u for p, u in reversed(cites) if p < m.start()]
        out.append((q, _unique(after + before)))
    return out


def _clauses(s: str, refs: dict[str, str]) -> list[str]:
    """Split a sentence at semicolons when more than one part cites its own source."""
    parts, last = [], 0
    for m in re.finditer(r";\s+", s):                 # only semicolons outside parentheses and brackets
        if not _depth(s[:m.start()]):
            parts.append(s[last:m.start()])
            last = m.end()
    parts.append(s[last:])
    if sum(1 for p in parts if _cites_at(p, refs)) < 2:
        return [s]
    out, buf = [], []
    for p in parts:
        buf.append(p)
        if _cites_at(p, refs):
            out.append("; ".join(buf))
            buf = []
    if buf:
        out[-1] += "; " + "; ".join(buf)
    return out


def _plain(s: str) -> str:
    """Claim text as a reader sees it: link text kept, URLs and markers dropped."""
    s = _INLINE.sub(lambda m: m.group(1), s)
    s = _AUTOLINK.sub("", s)
    s = _URL.sub("", s)
    s = _MARKER.sub("", s)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"[*_`]{1,3}", "", s)
    s = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip(" |")


def parse_markdown(text: str) -> list[Claim]:
    lines = text.splitlines()
    refs, skip = _references(lines)
    claims: list[Claim] = []
    segs = _segments(lines, skip)
    prev_urls: list[str] = []
    for k, (line, seg, is_bq) in enumerate(segs):
        if is_bq:
            urls = _citations(seg, refs) or list(prev_urls)
            if not urls and k + 1 < len(segs):
                urls = _citations(segs[k + 1][1], refs)
            quote = _plain(seg).strip("\"“”")
            if urls and len(quote.split()) >= MIN_QUOTE_WORDS:
                claims.append(Claim("", f"(block quote) {quote[:200]}", urls, [quote], [urls], line,
                                    note="" if _citations(seg, refs) else "link taken from the nearby paragraph"))
            continue
        is_row = " | " in seg                           # a table row is one claim: its cells belong together
        sents = [seg] if is_row else [c for s in _sentences(seg) for c in _clauses(s, refs)]
        cites = [_citations(s, refs) for s in sents]
        for j, s in enumerate(sents):
            urls, quoted, note = cites[j], _quotes(s, refs), ""
            if quoted and not urls:
                nearby = [c for c in cites[j + 1:] if c] or [c for c in cites[:j][::-1] if c]
                if nearby:
                    urls, note = nearby[0], "link taken from the same paragraph"
                    quoted = [(q, urls) for q, _ in quoted]
            if not urls:
                continue
            text = _plain(s)
            if len(text.split()) < MIN_CLAIM_WORDS and claims and claims[-1].line == line:
                prev = claims[-1]                     # a fragment: its links belong to the claim before
                prev.urls += [u for u in urls if u not in prev.urls]
                continue
            claims.append(Claim("", text, urls, [q for q, _ in quoted], [u for _, u in quoted], line, note))
        prev_urls = next((c for c in cites[::-1] if c), prev_urls)
    for i, c in enumerate(claims, 1):
        c.id = f"c{i}"
    return claims


# ---- JSON ----------------------------------------------------------------------------------

def _receipts_json(items: list) -> list[Claim]:
    out = []
    for i, c in enumerate(items, 1):
        if not isinstance(c, dict):
            continue
        url = str(c.get("url") or c.get("source_url") or c.get("link") or "")
        quote = str(c.get("quote") or "").strip()
        out.append(Claim(str(c.get("id") or f"c{i}"), str(c.get("claim") or c.get("text") or ""),
                         [url] if url else [], [quote] if quote else []))
    return out


def _openai(data: dict) -> list[Claim]:
    """Responses API: url_citation annotations index into the output text."""
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c)
    md_parts = []
    for c in parts:
        text, notes = c.get("text", ""), c.get("annotations", []) or []
        cites = sorted((a for a in notes if a.get("type") == "url_citation" and a.get("url")),
                       key=lambda a: a.get("end_index", 0), reverse=True)
        defs = []
        for n, a in enumerate(cites):
            label = f"oa{len(md_parts)}-{n}"
            end = min(int(a.get("end_index", len(text))), len(text))
            text = text[:end] + f" [{label}]" + text[end:]
            defs.append(f"[{label}]: {a['url']}")
        md_parts.append(text + "\n\n" + "\n".join(defs))
    return parse_markdown("\n\n".join(md_parts))


def _anthropic(data: dict) -> list[Claim]:
    """Messages API: each text block carries citations with the cited text taken from the page."""
    out = []
    for block in data.get("content", []):
        if block.get("type") != "text":
            continue
        cites = [c for c in (block.get("citations") or []) if c.get("url")]
        if not cites:
            continue
        text = block.get("text", "").strip()
        for c in cites:
            q = (c.get("cited_text") or "").strip()
            out.append(Claim("", text, [c["url"]], [q] if q else [], note="quote is Claude's cited_text"))
    for i, c in enumerate(out, 1):
        c.id = f"c{i}"
    return out


def _perplexity(data: dict) -> list[Claim]:
    msg = data["choices"][0]["message"]["content"]
    urls = data.get("citations") or [r.get("url") for r in data.get("search_results", []) if r.get("url")]
    defs = "\n".join(f"[{i}]: {u}" for i, u in enumerate(urls, 1))
    return parse_markdown(msg + "\n\n" + defs)


def parse_json(data) -> list[Claim]:
    if isinstance(data, list):
        return _receipts_json(data)
    if isinstance(data, dict):
        if isinstance(data.get("claims"), list):
            return _receipts_json(data["claims"])
        if isinstance(data.get("output"), list):
            return _openai(data)
        if isinstance(data.get("content"), list) and data.get("type") == "message":
            return _anthropic(data)
        if isinstance(data.get("choices"), list):
            return _perplexity(data)
    raise ValueError("JSON not recognised: expected {\"claims\": [...]}, an OpenAI Responses object, "
                     "an Anthropic message or a Perplexity chat completion")


def parse_report(raw: str) -> list[Claim]:
    s = raw.strip()
    if s.startswith("```"):                         # a JSON answer wrapped in a code fence
        s = re.sub(r"^```\w*\s*|\s*```$", "", s)
    if s[:1] in "{[":
        try:
            data = json.loads(s)
        except ValueError:
            m = re.search(r"\{.*\}", s, re.S)        # tolerate a code fence or stray text
            data = json.loads(m.group(0)) if m else None
        if data is not None:
            return parse_json(data)
    return parse_markdown(raw)
