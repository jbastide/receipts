"""Planted-defect benchmark for claim support.

Starts from claims in the Haiku, Sonnet or Opus study that both blind graders marked OK and whose
quote is word for word on the cited page. Each is checked as written (it should pass) and in four
damaged versions (each should be flagged):

  number     one number in the claim changed (30 -> 61; years moved by three)
  negation   the first auxiliary verb negated (may -> may not, can -> cannot)
  swap       the claim checked against another claim's page, on a different website
  overclaim  a specific invented fact appended ("according to a 2019 survey of 4,000 law firms")

A method "catches" a defect when its verdict is anything but supported, and "false-alarms" when it
flags an original.

Usage:
  uv run --extra support python bench/planted.py STUDY_DIR --cache DIR --out DIR \
      [--n 80] [--methods minicheck,hhem] [--judge-url URL --judge-model NAME]
Stages are saved to --out and resumed, so a long judge run can be stopped and restarted.
"""
import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from urllib.parse import urlparse

from receipts.fetch import Fetcher
from receipts.pages import read_page
from receipts.support import CHECKERS, Judge, SupportChecker

KINDS = ("original", "number", "negation", "swap", "overclaim")
EXTRAS = ["according to a 2019 survey of 4,000 law firms",
          "and the same rule has applied in every country since 2015",
          "with no exceptions for any business category",
          "and Google charges a separate fee of $25 for each recording"]
_NUM = re.compile(r"(?<![\w.$/-])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w/-])")
_AUX = re.compile(r"\b(may|can|will|must|should|does|do|is|are|was|were|has|have)\b")


def change_number(claim):
    m = _NUM.search(claim)
    if not m:
        return None
    v = int(m.group(1).replace(",", ""))
    new = v + 3 if 1900 <= v <= 2100 else (v + 4 if v < 10 else v * 2 + 1)
    return claim[:m.start()] + str(new) + claim[m.end():]


_NEGATIVE = re.compile(r"\b(not|never|no|none|nor|cannot)\b|n't\b", re.I)


def negate(claim):
    if _NEGATIVE.search(claim):                      # negating a negative claim makes nonsense ("may not not")
        return None
    m = _AUX.search(claim, 12)                       # skip a leading "It is" / "There are"
    if not m:
        return None
    rep = "cannot" if m.group(1) == "can" else m.group(1) + " not"
    return claim[:m.start()] + rep + claim[m.end():]


def build_cases(study, fetcher, n, seed=7, min_numbers=40):
    rows = list(csv.DictReader(open(os.path.join(study, "data.csv"), encoding="utf-8")))
    ok, seen = [], set()
    for r in rows:
        if r["quote_grade"] == "verbatim" and r["grader1"] == "OK" and r["grader2"] == "OK" and r["claim"] not in seen:
            seen.add(r["claim"])
            ok.append(r)
    random.Random(seed).shuffle(ok)
    picked = []
    for r in ok:
        if read_page(r["url"], fetcher).readable:
            picked.append(r)
        if len(picked) == n:
            break
    cases = []
    for i, r in enumerate(picked):
        base = {"base": f"{r['run']}/{r['claim_id']}", "url": r["url"]}
        cases.append({**base, "kind": "original", "claim": r["claim"]})
        if (c := change_number(r["claim"])):
            cases.append({**base, "kind": "number", "claim": c})
        if (c := negate(r["claim"])):
            cases.append({**base, "kind": "negation", "claim": c})
        host = urlparse(r["url"]).netloc
        others = [o for o in picked if urlparse(o["url"]).netloc != host]
        if others:
            other = others[i % len(others)]
            cases.append({**base, "kind": "swap", "claim": r["claim"], "url": other["url"]})
        cases.append({**base, "kind": "overclaim", "claim": r["claim"].rstrip(". ") + ", " + EXTRAS[i % len(EXTRAS)] + "."})
    # Few claims carry a number, so top the number set up to `min_numbers` from the rest of the pool;
    # each extra claim also enters as an original.
    have = sum(1 for c in cases if c["kind"] == "number")
    for r in ok:
        if have >= min_numbers:
            break
        if r in picked or not change_number(r["claim"]) or not read_page(r["url"], fetcher).readable:
            continue
        base = {"base": f"{r['run']}/{r['claim_id']}", "url": r["url"]}
        cases += [{**base, "kind": "original", "claim": r["claim"]},
                  {**base, "kind": "number", "claim": change_number(r["claim"])}]
        have += 1
    for k, c in enumerate(cases):
        c["id"] = k
    return cases


def run_method(name, checker, cases, fetcher, path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            done[d["id"]] = d
    with open(path, "a", encoding="utf-8") as fh:
        for c in cases:
            if c["id"] in done:
                continue
            t = time.time()
            s = checker.assess(c["claim"], [], c["url"], read_page(c["url"], fetcher).text)
            d = {"id": c["id"], "verdict": s.verdict, "score": s.checker_score, "judge": s.judge_verdict,
                 "evidence": s.evidence[:2], "reason": s.reason, "seconds": round(time.time() - t, 2)}
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            done[c["id"]] = d
            print(f"\r{name}: {len(done)}/{len(cases)}", end="", flush=True)
    print()
    return done


def usable(case, by_base):
    """Cases made before negate() skipped negative claims are left out of the scores."""
    if case["kind"] != "negation":
        return True
    original = by_base.get(case["base"], "")
    return not _NEGATIVE.search(original)


def summarize(cases, results):
    lines = ["| Method | Originals passed | " + " | ".join(f"Caught: {k}" for k in KINDS[1:]) +
             " | Caught: all defects | Seconds per claim |",
             "| --- " * (len(KINDS) + 3) + "|"]
    counts = {k: sum(1 for c in cases if c["kind"] == k) for k in KINDS}
    for name, res in results.items():
        flagged = defaultdict(int)
        for c in cases:
            if res[c["id"]]["verdict"] != "supported":
                flagged[c["kind"]] += 1
        n_def = sum(counts[k] for k in KINDS[1:])
        caught = sum(flagged[k] for k in KINDS[1:])
        secs = sum(res[c["id"]].get("seconds", 0) for c in cases) / len(cases)
        cells = [f"{counts['original'] - flagged['original']} of {counts['original']} "
                 f"({100 * (counts['original'] - flagged['original']) / counts['original']:.0f}%)"]
        cells += [f"{flagged[k]} of {counts[k]} ({100 * flagged[k] / counts[k]:.0f}%)" for k in KINDS[1:]]
        cells.append(f"{caught} of {n_def} ({100 * caught / n_def:.0f}%)")
        cells.append(f"{secs:.1f}" if secs else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def combine(a, b):
    """Flag a claim if either method flags it."""
    return {i: {"verdict": "supported" if a[i]["verdict"] == "supported" and b[i]["verdict"] == "supported"
                else "not_supported", "seconds": a[i].get("seconds", 0) + b[i].get("seconds", 0)} for i in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--methods", default="minicheck,hhem")
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--judge-model", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fetcher = Fetcher(a.cache, offline=True)

    cpath = os.path.join(a.out, "cases.jsonl")
    if os.path.exists(cpath):
        cases = [json.loads(l) for l in open(cpath, encoding="utf-8")]
    else:
        cases = build_cases(a.study, fetcher, a.n)
        with open(cpath, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(c) + "\n" for c in cases)

    names = [m for m in a.methods.split(",") if m]
    for m in names:
        run_method(m, SupportChecker(CHECKERS[m]()), cases, fetcher, os.path.join(a.out, f"scores-{m}.jsonl"))
    if a.judge_url:
        jname = "judge-" + (a.judge_model or "local")
        names.append(jname)
        run_method(jname, SupportChecker(None, Judge(a.judge_url, a.judge_model)), cases, fetcher,
                   os.path.join(a.out, f"scores-{jname}.jsonl"))

    results = {}
    for f in sorted(os.listdir(a.out)):
        if f.startswith("scores-") and f.endswith(".jsonl"):
            name = f[len("scores-"):-len(".jsonl")]
            res = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.out, f), encoding="utf-8")}
            if all(c["id"] in res for c in cases):
                results[name] = res
    judges = [k for k in results if k.startswith("judge-")]
    for c in [k for k in results if not k.startswith("judge-")]:
        for j in judges:
            results[f"{c} or {j}"] = combine(results[c], results[j])
    by_base = {c["base"]: c["claim"] for c in cases if c["kind"] == "original"}
    kept = [c for c in cases if usable(c, by_base)]
    dropped = len(cases) - len(kept)
    table = summarize(kept, results)
    if dropped:
        table += f"\n\n{dropped} negation cases made from claims that were already negative were left out."
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
