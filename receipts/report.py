"""Write a check result as Markdown (to read), JSON (everything) and CSV (one row per quote)."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from .check import LinkResult, Result
from .text import squash

LIMITS = """## What this check does not do

- It does not judge whether a page supports a claim. A quote can be on the page word for word and
  still not back the claim made with it, and claims without quotes are only checked for a working link.
- A page that changed after the report was written can make a correct quote look missing. The saved
  copy (see Pages read) is what this check saw.
- "No Wayback capture" does not prove a link was invented: the Wayback Machine doesn't have every page.
- Quoted phrases under five words are not checked; they are usually terms, not quotations.
- "Unreadable" means we could not see the page's text (it refused the request, or builds its text with
  JavaScript). The quote may well be there; check it by hand.
"""


def _cell(s: str, limit: int = 0) -> str:
    return squash(s, limit).replace("|", "\\|")


def _link_what(l: LinkResult) -> str:
    what = {"ok": "read", "unreadable": "too little text to check (script-rendered or blocked)",
            "refused": f"refused our request ({l.status})", "gone": f"not found ({l.status})",
            "error": f"no response ({l.error or l.status})"}[l.state]
    if l.via == "arctic-shift":
        what += ", read from the Arctic Shift Reddit archive"
    return what


def _archive(l: LinkResult) -> str:
    a = l.archive
    if a is None:
        return ""
    if a.state == "archived":
        d = a.first_capture
        return f"first captured {d[:4]}-{d[4:6]}-{d[6:8]}: the page existed, then moved or was removed"
    if a.state == "none":
        return "no capture: the page may never have existed"
    return "Wayback lookup failed"


def markdown(r: Result, cache_dir: str = "") -> str:
    out = [f"# Receipts check: {os.path.basename(r.source) or 'report'}", "",
           f"Checked {r.checked_at.replace('T', ' ').replace('+00:00', '')} UTC.", "",
           f"**{r.summary()}**", ""]

    bad = [(c, q) for c in r.claims for q in c.quotes if q.grade in ("missing", "edited")]
    bad.sort(key=lambda cq: (cq[1].grade != "missing", cq[1].share))
    if bad:
        out += ["## Quotes to look at", ""]
        for c, q in bad:
            link = r.links[q.url]
            out += [f"### {c.claim.id}: {q.grade}" + (f" (line {c.claim.line})" if c.claim.line else ""), "",
                    f"- **Claim:** {squash(c.claim.text, 400)}",
                    f"- **Quote:** “{squash(q.quote, 600)}”",
                    f"- **Cited page:** <{q.url}>, {_link_what(link)}"]
            if q.closest:
                out.append(f"- **Nearest passage on the page** ({round(q.share * 100)}% of the quote's "
                           f"five-word runs found): “{squash(q.closest, 600)}”")
            else:
                out.append("- **Nearest passage on the page:** none of the quote's five-word runs were found")
            if c.claim.note:
                out.append(f"- Note: {c.claim.note}")
            out.append("")

    unread = [(c, q) for c in r.claims for q in c.quotes if q.grade == "unreadable"]
    if unread:
        out += ["## Quotes that could not be checked", "",
                "| Claim | Quote | Page | Why |", "| --- | --- | --- | --- |"]
        for c, q in unread:
            out.append(f"| {c.claim.id} | “{_cell(q.quote, 120)}” | {_cell(q.url, 80)} | "
                       f"{_cell(_link_what(r.links[q.url]))} |")
        out.append("")

    problem_links = [l for l in r.links.values() if l.state != "ok"]
    if problem_links:
        out += ["## Links to look at", "", "| Link | What happened | Wayback Machine |",
                "| --- | --- | --- |"]
        order = {"gone": 0, "error": 1, "refused": 2, "unreadable": 3}
        for l in sorted(problem_links, key=lambda l: (order[l.state], l.url)):
            out.append(f"| {_cell(l.url, 100)} | {_cell(_link_what(l))} | {_cell(_archive(l))} |")
        out.append("")

    out += ["## Every claim", "", "| # | Claim | Quote | Link |", "| --- | --- | --- | --- |"]
    for c in r.claims:
        grades = ", ".join(q.grade for q in c.quotes) or "no quote"
        states = ", ".join(f"{r.links[u].state} ({r.links[u].host})" for u in c.claim.urls)
        out.append(f"| {c.claim.id} | {_cell(c.claim.text, 140)} | {grades} | {_cell(states, 120)} |")
    out.append("")

    out += ["## Pages read", ""]
    if cache_dir:
        out += [f"Saved copies are in `{cache_dir}`, named by the first 32 characters of the SHA-256 "
                "of each URL.", ""]
    out += ["| Link | Status | Fetched (UTC) | SHA-256 of the body |", "| --- | --- | --- | --- |"]
    for l in sorted(r.links.values(), key=lambda l: l.url):
        status = f"{l.status or '-'}" + (" (Arctic Shift archive)" if l.via == "arctic-shift" else "")
        out.append(f"| {_cell(l.url, 100)} | {status} | {l.fetched_at.replace('+00:00', '')} | "
                   f"{l.sha256[:16] or '-'} |")
    out += ["", LIMITS]
    return "\n".join(out)


def as_dict(r: Result) -> dict:
    return {"source": r.source, "checked_at": r.checked_at, "summary": r.summary(),
            "quote_counts": dict(r.quote_counts()), "link_counts": dict(r.link_counts()),
            "claims": [{**asdict(c.claim), "quotes": [asdict(q) for q in c.quotes]} for c in r.claims],
            "links": {u: asdict(l) for u, l in r.links.items()}}


CSV_FIELDS = ["claim_id", "line", "claim", "quote", "quote_grade", "shingle_share", "url",
              "link_state", "status", "wayback", "first_capture", "note"]


def rows(r: Result) -> list[dict]:
    out = []
    for c in r.claims:
        items = c.quotes or [None]
        for q in items:
            url = q.url if q else (c.claim.urls[0] if c.claim.urls else "")
            l = r.links.get(url)
            out.append({"claim_id": c.claim.id, "line": c.claim.line, "claim": c.claim.text,
                        "quote": q.quote if q else "", "quote_grade": q.grade if q else "",
                        "shingle_share": q.share if q else "", "url": url,
                        "link_state": l.state if l else "", "status": l.status if l else "",
                        "wayback": l.archive.state if l and l.archive else "",
                        "first_capture": l.archive.first_capture if l and l.archive else "",
                        "note": c.claim.note})
    return out


def write_all(r: Result, out_dir: str, cache_dir: str = "") -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"check.{k}") for k in ("md", "json", "csv")}
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write(markdown(r, cache_dir))
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(as_dict(r), fh, indent=1, ensure_ascii=False)
    with open(paths["csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows(r))
    return paths
