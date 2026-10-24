"""Write a check result as Markdown (to read), JSON (everything) and CSV (one row per quote)."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from .check import LinkResult, Result
from .text import squash

NO_SUPPORT = """- It does not judge whether a page supports a claim (run with --support for that). A quote can be on
  the page word for word and still not back the claim made with it, and claims without quotes are only
  checked for a working link.
"""

WITH_SUPPORT = """- Support verdicts come from small models reading the sentences of the page that best match the
  claim. They are a reason to look, not a ruling: read the evidence sentences, and the page, before
  deciding. A claim drawn from a part of the page the search did not pick up can be marked unsupported.
"""

LIMITS = """## What this check does not do

{support}- A page that changed after the report was written can make a correct quote look missing. The saved
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
           f"Checked {r.checked_at.replace('T', ' ').replace('+00:00', '')} UTC." + _method(r), "",
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

    out += _support_sections(r)

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

    has_support = any(c.support for c in r.claims)
    head = "| # | Claim | Quote | Link |" + (" Support |" if has_support else "")
    out += ["## Every claim", "", head, "| --- " * head.count(" |") + "|"]
    for c in r.claims:
        grades = ", ".join(q.grade for q in c.quotes) or "no quote"
        states = ", ".join(f"{r.links[u].state} ({r.links[u].host})" for u in c.claim.urls)
        row = f"| {c.claim.id} | {_cell(c.claim.text, 140)} | {grades} | {_cell(states, 120)} |"
        if has_support:
            row += f" {_support_cell(c.support)} |"
        out.append(row)
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
    out += ["", LIMITS.format(support=WITH_SUPPORT if has_support else NO_SUPPORT)]
    return "\n".join(out)


def _method(r: Result) -> str:
    m = r.method
    if not m:
        return ""
    parts = [p for p in (m.get("checker") and f"checker model {m['checker']}",
                         m.get("judge") and f"judge {m['judge'].removeprefix('judge:')}") if p]
    return " Support checked by " + " and ".join(parts) + "." if parts else ""


def _support_cell(s) -> str:
    if s is None:
        return ""
    label = s.verdict.replace("_", " ")
    if s.facts:
        ok = sum(1 for f in s.facts if f["verdict"] == "supported")
        return label + f" ({ok} of {len(s.facts)} facts)" + (" ⚑" if s.flag else "")
    if s.checker_score is not None:
        label += f" ({s.checker_score:.2f})"
    return label + (" ⚑" if s.flag else "")


def _support_sections(r: Result) -> list[str]:
    out = []
    weak = [c for c in r.claims if c.support and c.support.verdict in ("not_supported", "partly")]
    weak.sort(key=lambda c: (c.support.verdict != "not_supported", c.support.checker_score or 0))
    if weak:
        out += ["## Claims the cited page doesn't support", ""]
        for c in weak:
            s = c.support
            out += [f"### {c.claim.id}: {s.verdict.replace('_', ' ')}" + (f" (line {c.claim.line})" if c.claim.line else ""), "",
                    f"- **Claim:** {squash(c.claim.text, 500)}",
                    f"- **Cited page:** <{s.url}>"]
            out += _support_detail(s)
            out.append("")
    flagged = [c for c in r.claims if c.support and c.support.flag and c.support.verdict not in ("not_supported", "partly")]
    if flagged:
        out += ["## Check by hand", "", "The two models disagree on these, or the judge was unsure.", ""]
        for c in flagged:
            out += [f"### {c.claim.id}: {c.support.verdict.replace('_', ' ')}", "",
                    f"- **Claim:** {squash(c.claim.text, 500)}", f"- **Cited page:** <{c.support.url}>"]
            out += _support_detail(c.support)
            out.append("")
    return out


def _support_detail(s) -> list[str]:
    out = []
    if s.reason:
        out.append(f"- **Why:** {squash(s.reason, 300)}")
    opinions = []
    if s.judge_verdict:
        opinions.append(f"judge: {s.judge_verdict.replace('_', ' ')}")
    if s.checker_score is not None:
        opinions.append(f"checker model: {s.checker_score:.2f} (0 = no support, 1 = support)")
    if opinions:
        out.append("- **Opinions:** " + "; ".join(opinions))
    if s.flag:
        out.append(f"- **Look closer:** {s.flag}")
    if s.facts:
        out.append("- **The claim's facts, checked one by one:**")
        for i, f in enumerate(s.facts, 1):
            mark = {"supported": "✓", "partly": "◐", "not_supported": "✗"}.get(f["verdict"], "?")
            out.append(f"  {i}. {mark} {squash(f['text'], 300)} ({f['verdict'].replace('_', ' ')})")
            if f["verdict"] != "supported" and f.get("reason"):
                out.append(f"     - {squash(f['reason'], 250)}")
            if f.get("evidence"):
                out.append(f"     - closest on the page: “{squash(f['evidence'][0], 300)}”")
        return out
    if s.evidence:
        out.append("- **Closest sentences on the page:**")
        out += [f"  - “{squash(e, 400)}”" for e in s.evidence]
    return out


def as_dict(r: Result) -> dict:
    return {"source": r.source, "checked_at": r.checked_at, "summary": r.summary(), "method": r.method,
            "quote_counts": dict(r.quote_counts()), "link_counts": dict(r.link_counts()),
            "claims": [{**asdict(c.claim), "quotes": [asdict(q) for q in c.quotes],
                        "support": asdict(c.support) if c.support else None} for c in r.claims],
            "links": {u: asdict(l) for u, l in r.links.items()}}


CSV_FIELDS = ["claim_id", "line", "claim", "quote", "quote_grade", "shingle_share", "url",
              "link_state", "status", "wayback", "first_capture", "support", "checker_score",
              "judge_verdict", "facts_supported", "support_flag", "support_url", "note"]


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
                        "support": c.support.verdict if c.support else "",
                        "checker_score": "" if not c.support or c.support.checker_score is None else c.support.checker_score,
                        "judge_verdict": c.support.judge_verdict if c.support else "",
                        "facts_supported": (f"{sum(1 for f in c.support.facts if f['verdict'] == 'supported')}"
                                            f"/{len(c.support.facts)}") if c.support and c.support.facts else "",
                        "support_flag": c.support.flag if c.support else "",
                        "support_url": c.support.url if c.support else "",
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
