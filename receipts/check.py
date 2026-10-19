"""Check every claim in a report: fetch what it cites, grade its quotes, test its links."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from .fetch import Fetcher
from .grade import QUOTE_GRADES, Archive, grade_quote, link_state, wayback
from .pages import Page, read_page
from .parse import Claim
from .support import Support, SupportChecker

# When a quote cites several pages, report the least bad result. "unreadable" ranks above "missing":
# a quote absent from one page may still be on another page we could not read.
_RANK = {"verbatim": 0, "elided": 1, "edited": 2, "unreadable": 3, "missing": 4}


@dataclass
class QuoteResult:
    quote: str
    grade: str
    url: str
    share: float = 0.0
    closest: str = ""


@dataclass
class LinkResult:
    url: str
    state: str
    status: int
    via: str = "direct"
    sha256: str = ""
    fetched_at: str = ""
    error: str = ""
    archive: Archive | None = None

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc


@dataclass
class ClaimResult:
    claim: Claim
    quotes: list[QuoteResult] = field(default_factory=list)
    support: Support | None = None


@dataclass
class Result:
    source: str
    checked_at: str
    claims: list[ClaimResult]
    links: dict[str, LinkResult]
    method: dict = field(default_factory=dict)

    def quote_counts(self) -> Counter:
        c = Counter({g: 0 for g in QUOTE_GRADES})
        c.update(q.grade for r in self.claims for q in r.quotes)
        return c

    def link_counts(self) -> Counter:
        c = Counter(l.state for l in self.links.values())
        for l in self.links.values():
            if l.archive:
                c["gone_" + l.archive.state] += 1
        return c

    def summary(self) -> str:
        q, l = self.quote_counts(), self.link_counts()
        n_q = sum(q.values())
        parts = [f"{len(self.claims)} claims citing {len(self.links)} links"]
        if n_q:
            bits = [f"{q['verbatim']} word for word"]
            bits += [f"{q[g]} {g}" for g in ("elided", "edited", "missing", "unreadable") if q[g]]
            parts.append(f"quotes: {n_q} ({', '.join(bits)})")
        else:
            parts.append("no quotes to check")
        bad = [f"{l[s]} {s}" for s in ("unreadable", "refused", "gone", "error") if l[s]]
        parts.append("links: " + (f"{l['ok']} read" + (", " + ", ".join(bad) if bad else "")))
        never = l["gone_none"]
        if never:
            parts.append(f"{never} with no Wayback record")
        s = self.support_counts()
        if sum(s.values()):
            bits = [f"{s[v]} {v.replace('_', ' ')}" for v in SUPPORT_ORDER if s[v]]
            flagged = sum(1 for r in self.claims if r.support and r.support.flag)
            parts.append("support: " + ", ".join(bits) + (f" ({flagged} to check by hand)" if flagged else ""))
        return "; ".join(parts) + "."

    def support_counts(self) -> Counter:
        return Counter(r.support.verdict for r in self.claims if r.support)


SUPPORT_ORDER = ("supported", "partly", "not_supported", "not_checked")
MAX_PAGES_PER_CLAIM = 3


def assess_claim(c: Claim, pages: dict[str, Page], support: SupportChecker) -> Support:
    """Support from the best of the claim's readable pages (at most three)."""
    readable = [u for u in c.urls if pages[u].readable][:MAX_PAGES_PER_CLAIM]
    if not readable:
        return Support(c.urls[0] if c.urls else "", "not_checked", reason="no cited page could be read")
    best = None
    for u in readable:
        s = support.assess(c.text, c.quotes, u, pages[u].text)
        if best is None or SUPPORT_ORDER.index(s.verdict) < SUPPORT_ORDER.index(best.verdict):
            best = s
        if best.verdict == "supported":
            break
    return best


def check(claims: list[Claim], fetcher: Fetcher, *, source: str = "", workers: int = 8,
          use_wayback: bool = True, support: SupportChecker | None = None, progress=None) -> Result:
    urls = sorted({u for c in claims for u in c.urls})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pages: dict[str, Page] = dict(zip(urls, pool.map(lambda u: read_page(u, fetcher), urls)))

    links: dict[str, LinkResult] = {}
    for u, p in pages.items():
        links[u] = LinkResult(u, link_state(p), p.status, p.via, p.sha256, p.fetched_at, p.error)
    if use_wayback:
        lost = [u for u, l in links.items() if l.state in ("gone", "error") and u.startswith("http")]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
            for u, a in zip(lost, pool.map(lambda u: wayback(u, fetcher), lost)):
                links[u].archive = a

    results = []
    for c in claims:
        r = ClaimResult(c)
        for i, q in enumerate(c.quotes):
            best = None
            order = c.quote_urls[i] if i < len(c.quote_urls) and c.quote_urls[i] else c.urls
            for u in order:                         # nearest link first; it wins a tie
                g = grade_quote(q, pages[u])
                cand = QuoteResult(q, g.grade, u, g.share, g.closest)
                if best is None or _RANK[cand.grade] < _RANK[best.grade] or (
                        cand.grade == best.grade and cand.share > best.share):
                    best = cand
            if best is not None:
                r.quotes.append(best)
        if support is not None:
            r.support = assess_claim(c, pages, support)
            if progress:
                progress(len(results) + 1, len(claims))
        results.append(r)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    method = {}
    if support is not None:
        method = {"checker": getattr(support.checker, "name", "") if support.checker else "",
                  "judge": support.judge.name if support.judge else "",
                  "threshold": support.threshold}
    return Result(source, now, results, links, method)
