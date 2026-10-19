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


def make_support(a: argparse.Namespace):
    if not (a.support or a.judge_url):
        return None
    from .support import CHECKERS, Judge, SupportChecker
    checker = None
    if a.checker != "none":
        try:
            checker = CHECKERS[a.checker]()
        except ImportError:
            sys.exit("The checker models need the optional extra: uv sync --extra support "
                     "(or use --checker none with --judge-url)")
    judge = Judge(a.judge_url, a.judge_model, samples=a.judge_samples,
                  api_key=os.environ.get("RECEIPTS_JUDGE_KEY", "")) if a.judge_url else None
    if checker is None and judge is None:
        sys.exit("--checker none needs --judge-url")
    return SupportChecker(checker, judge, threshold=a.threshold)


def _progress(i: int, n: int):
    if sys.stderr.isatty():
        print(f"\r  support: {i}/{n} claims", end="" if i < n else "\n", file=sys.stderr, flush=True)


def cmd_check(a: argparse.Namespace) -> int:
    fetcher = Fetcher(a.cache, offline=a.offline, timeout=a.timeout, user_agent=a.user_agent)
    support = make_support(a)
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
        result = check(claims, fetcher, source=path, workers=a.workers, use_wayback=not a.no_wayback,
                       support=support, progress=_progress)
        out_dir = a.out if len(a.reports) == 1 and a.out else os.path.join(a.out or ".", f"{_stem(path)}-check")
        paths = write_all(result, out_dir, a.cache)
        print(f"{path}: {result.summary()}")
        print(f"  report: {paths['md']}")
        counts = result.quote_counts() + result.link_counts() + result.support_counts()
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
                   help="exit with status 1 if any quote, link or claim has one of these comma-separated "
                        "results, e.g. missing,edited,gone,not_supported")
    s = c.add_argument_group("claim support", "does the cited page support each claim? (step 2)")
    s.add_argument("--support", action="store_true", help="check claim support with a small checker model")
    s.add_argument("--checker", choices=["minicheck", "hhem", "none"], default="minicheck",
                   help="checker model (default: %(default)s); needs: uv sync --extra support")
    s.add_argument("--judge-url", default=os.environ.get("RECEIPTS_JUDGE_URL", ""),
                   help="OpenAI-compatible endpoint of a local judge model, e.g. http://127.0.0.1:8080/v1 "
                        "(llama.cpp, Ollama, vLLM, LM Studio); implies --support")
    s.add_argument("--judge-model", default=os.environ.get("RECEIPTS_JUDGE_MODEL", ""),
                   help="model name to send to the endpoint, if it serves several")
    s.add_argument("--judge-samples", type=int, default=1,
                   help="ask the judge this many times and take the majority (default: %(default)s)")
    s.add_argument("--threshold", type=float, default=0.5,
                   help="checker score at or above which a claim counts as supported (default: %(default)s)")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
