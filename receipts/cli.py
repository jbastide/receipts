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
            device = a.checker_device
            if device == "auto" and a.judge_url and sys.platform == "darwin":
                device = "cpu"                          # see support._device
            checker = CHECKERS[a.checker](device=device)
        except ImportError:
            sys.exit("The checker models need the optional extra: uv sync --extra support "
                     "(or use --checker none with --judge-url)")
    judge = Judge(a.judge_url, a.judge_model, samples=a.judge_samples,
                  api_key=os.environ.get("RECEIPTS_JUDGE_KEY", "")) if a.judge_url else None
    if checker is None and judge is None:
        sys.exit("--checker none needs --judge-url")
    return SupportChecker(checker, judge, threshold=a.threshold, split=not a.no_split)


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


def cmd_study(a: argparse.Namespace) -> int:
    from . import study as st
    s = st.load_study(a.study)
    if a.action == "collect":
        from .collect import reddit_threads
        ids = [l.strip() for l in open(a.reddit_ids, encoding="utf-8") if l.strip()]
        items = reddit_threads(ids, Fetcher(a.cache), pause=a.pause, progress=_progress_label("collect"))
        st.write_items(items, s.items_path)
        units = sum(len(i["units"]) for i in items)
        print(f"{len(items)} of {len(ids)} items, {units} units -> {s.items_path}")
        return 0 if len(items) == len(ids) else 1
    if a.action == "code":
        cfg = s.coders.get(a.coder)
        if not cfg:
            sys.exit(f"no [coders.{a.coder}] in {a.study}")
        coder = st.Coder(a.coder, cfg["url"], cfg.get("model", ""))
        ids = None
        if a.ids == "calibration":
            ids = [l.strip() for l in open(s.calibration_ids_path, encoding="utf-8") if l.strip()]
        elif a.ids:
            ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()]
        path = st.code_study(s, coder, limit=a.limit, progress=_progress_label(f"code {a.coder}"), ids=ids, tag=a.tag)
        rows = list(st.read_codes(path).values())
        failed = sum(r["failed_chunks"] for r in rows)
        dropped = sum(r["dropped_codes"] for r in rows)
        print(f"{len(rows)} items coded by {a.coder}; {failed} chunks without a usable answer; "
              f"{dropped} codes dropped for citing a unit not shown -> {path}")
        return 0
    if a.action == "verify":
        cfg = s.coders.get(a.coder)
        if not cfg or not a.source:
            sys.exit("verify needs --coder NAME and --source CODES (e.g. gemma, or gemma-v1)")
        coder = st.Coder(a.coder, cfg["url"], cfg.get("model", ""))
        path = st.verify_study(s, coder, a.source, progress=_progress_label(f"verify {a.source}"))
        rows = list(st.read_codes(path).values())
        print(f"{len(rows)} items verified; {sum(r['verify_questions'] for r in rows)} codes checked, "
              f"{sum(r['verify_rejected'] for r in rows)} rejected -> {path}")
        return 0
    if a.action == "calibrate":
        for name in a.coders:
            cal, held = st.calibrate(s, name)
            print(f"\n{name}: kappa against gold (pass at {s.min_kappa} on {cal[0]['n']} calibration items; "
                  f"held-out n={held[0]['n']})")
            print(f"  {'field':22} {'calib.':>7} {'pass':>5} {'held-out':>9}")
            for c, h in zip(cal, held):
                k = lambda v: "-" if v is None else f"{v:.2f}"
                print(f"  {c['field']:22} {k(c['kappa']):>7} {'yes' if c['passes'] else 'no':>5} {k(h['kappa']):>9}")
        return 0
    if a.action == "compare":
        rows = st.compare(s, a.coders[0], a.coders[1])
        print(f"{a.coders[0]} vs {a.coders[1]}, n={rows[0]['n'] if rows else 0}")
        for r in rows:
            print(f"  {r['field']:22} agree {r['agree_pct']:>5}%  kappa {'-' if r['kappa'] is None else r['kappa']}")
        return 0
    return 2


def _progress_label(label: str):
    def show(i: int, n: int):
        if sys.stderr.isatty():
            print(f"\r  {label}: {i}/{n}", end="" if i < n else "\n", file=sys.stderr, flush=True)
    return show


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
    s.add_argument("--checker-device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                   help="where the checker model runs (default: auto; on a Mac, the CPU when a judge is used)")
    s.add_argument("--judge-url", default=os.environ.get("RECEIPTS_JUDGE_URL", ""),
                   help="OpenAI-compatible endpoint of a local judge model, e.g. http://127.0.0.1:8080/v1 "
                        "(llama.cpp, Ollama, vLLM, LM Studio); implies --support")
    s.add_argument("--judge-model", default=os.environ.get("RECEIPTS_JUDGE_MODEL", ""),
                   help="model name to send to the endpoint, if it serves several")
    s.add_argument("--judge-samples", type=int, default=1,
                   help="ask the judge this many times and take the majority (default: %(default)s)")
    s.add_argument("--no-split", action="store_true",
                   help="judge long or many-figure claims whole instead of splitting them into single facts "
                        "(splitting needs --judge-url)")
    s.add_argument("--threshold", type=float, default=0.5,
                   help="checker score at or above which a claim counts as supported (default: %(default)s)")
    t = sub.add_parser("study", help="code study items against a codebook with local models (step 3)",
                       description="collect items, code them with a local model, measure each model "
                                   "against gold labels, and compare two coders")
    t.add_argument("action", choices=["collect", "code", "verify", "calibrate", "compare"])
    t.add_argument("study", help="the study's TOML file")
    t.add_argument("--coder", default="", help="code: which [coders.NAME] to use")
    t.add_argument("--coders", nargs="+", default=[], help="calibrate: coder names; compare: exactly two")
    t.add_argument("--limit", type=int, default=0, help="code: only the first N items")
    t.add_argument("--ids", default="", help="code: only these items: a file of ids, or 'calibration'")
    t.add_argument("--source", default="", help="verify: the codes to check, e.g. gemma or gemma-v1")
    t.add_argument("--tag", default="", help="code: write codes-CODER-TAG.jsonl instead (trying a prompt)")
    t.add_argument("--reddit-ids", default="", help="collect: a file with one Reddit thread id per line")
    t.add_argument("--cache", default=default_cache(), help="collect: where fetched pages are kept")
    t.add_argument("--pause", type=float, default=1.0, help="collect: seconds between live archive requests")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "study":
        return cmd_study(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
