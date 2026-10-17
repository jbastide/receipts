"""receipts check REPORT... : is every quote on the page it cites, and does every link exist?"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .check import check
from .fetch import DEFAULT_UA, Fetcher
from .parse import parse_report
from .report import write_all


def default_cache() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "receipts", "pages")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] or "report"


def cmd_check(a: argparse.Namespace) -> int:
    fetcher = Fetcher(a.cache, offline=a.offline, timeout=a.timeout, user_agent=a.user_agent)
    fail_on = {g.strip() for g in a.fail_on.split(",") if g.strip()}
    failed = False
    for path in a.reports:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        try:
            claims = parse_report(raw)
        except ValueError as e:
            print(f"{path}: {e}", file=sys.stderr)
            failed = True
            continue
        if not claims:
            print(f"{path}: no claims with links found")
            continue
        result = check(claims, fetcher, source=path, workers=a.workers, use_wayback=not a.no_wayback)
        out_dir = a.out if len(a.reports) == 1 and a.out else os.path.join(a.out or ".", f"{_stem(path)}-check")
        paths = write_all(result, out_dir, a.cache)
        print(f"{path}: {result.summary()}")
        print(f"  report: {paths['md']}")
        counts = result.quote_counts() + result.link_counts()
        if any(counts[g] for g in fail_on):
            failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="receipts", description=__doc__)
    p.add_argument("--version", action="version", version=f"receipts {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check the quotes and links in one or more research reports",
                       description="Fetch every page a report cites, keep a copy, grade each quote "
                                   "(verbatim, elided, edited, missing, unreadable) and test each link.")
    c.add_argument("reports", nargs="+", help="Markdown, text or JSON report files")
    c.add_argument("-o", "--out", default="",
                   help="output folder (default: <report name>-check in the current folder)")
    c.add_argument("--cache", default=default_cache(), help="where fetched pages are kept (default: %(default)s)")
    c.add_argument("--offline", action="store_true", help="use only pages already in the cache")
    c.add_argument("--no-wayback", action="store_true", help="don't look up dead links in the Wayback Machine")
    c.add_argument("--workers", type=int, default=8, help="pages fetched at once (default: %(default)s)")
    c.add_argument("--timeout", type=float, default=40.0, help="seconds per request (default: %(default)s)")
    c.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent header sent with each request")
    c.add_argument("--fail-on", default="",
                   help="exit with status 1 if any quote or link has one of these comma-separated "
                        "results, e.g. missing,edited,gone")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
