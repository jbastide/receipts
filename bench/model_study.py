"""Reproduce the quote checks of the Haiku, Sonnet or Opus research study, plus a negative control.

Usage: uv run python bench/model_study.py <path to haiku-vs-sonnet-vs-opus-research-study> [--cache DIR]

1. Grades every quote in runs/*.json and compares with the published grade in data.csv.
2. Negative control (as the study's control.py): one word in each quote of six words or more is
   replaced, and each quote is checked against a page it was not taken from. Neither may pass as verbatim.
Pages that changed since October 13, 2026 can make a published quote look edited or missing.
"""
import argparse
import csv
import glob
import os
import random
from collections import Counter

from receipts.check import check
from receipts.fetch import Fetcher
from receipts.grade import grade_quote
from receipts.pages import read_page
from receipts.parse import parse_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/receipts/pages"))
    a = ap.parse_args()
    fetcher = Fetcher(a.cache)

    published = {(r["run"], r["claim_id"]): r["quote_grade"]
                 for r in csv.DictReader(open(os.path.join(a.study, "data.csv"), encoding="utf-8"))}
    ours, pairs = Counter(), []
    for path in sorted(glob.glob(os.path.join(a.study, "runs", "*.json"))):
        run = os.path.basename(path)[:-5]
        result = check(parse_report(open(path, encoding="utf-8").read()), fetcher, source=path)
        for c in result.claims:
            for q in c.quotes:
                ours[(published.get((run, c.claim.id), "?"), q.grade)] += 1
                pairs.append((q.quote, q.url))

    print("published grade -> receipts grade: count")
    for (pub, got), n in sorted(ours.items()):
        print(f"  {pub:>9} -> {got:<10} {n}")
    same = sum(n for (pub, got), n in ours.items() if pub == got)
    print(f"agree: {same} of {sum(ours.values())}")

    random.seed(7)
    swap = cross = n = 0
    for quote, url in pairs:
        w = quote.split()
        if len(w) < 6:
            continue
        n += 1
        w[random.randrange(1, len(w) - 1)] = "zebra"
        swap += grade_quote(" ".join(w), read_page(url, fetcher)).grade == "verbatim"
        other = random.choice([p for p in pairs if p[1] != url])
        cross += grade_quote(other[0], read_page(url, fetcher)).grade == "verbatim"
    print(f"negative control: {n} quotes; accepted after one word replaced: {swap}; "
          f"accepted against a different page: {cross}")


if __name__ == "__main__":
    main()
