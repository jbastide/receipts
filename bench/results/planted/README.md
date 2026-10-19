# Planted-defect benchmark, October 19, 2026

Made by `bench/planted.py` from the Haiku, Sonnet or Opus study. The starting claims are ones both
blind graders marked OK and whose quote is word for word on the cited page. 80 were sampled (seed 7),
plus 25 more that contain a number. Pages were read from the copies saved on October 17, 2026.

Each claim is checked as written (it should pass) and in up to four damaged versions (each should be
flagged): a number changed, the first auxiliary verb negated, the claim checked against another
website's page, and a specific invented fact appended. 17 negation cases built from claims that were
already negative ("may not not hear") are left out.

Machine: MacBook Air, Apple M5, 16 GB. Checker models on the Apple GPU (MPS) in float32. Judge:
Gemma 4 12B instruction-tuned, QAT Q4_0 GGUF, llama.cpp `llama-server` build 11669, one slot,
temperature 0, JSON-schema output, reasoning off.

| Method | Originals passed | Caught: number | Caught: negation | Caught: swap | Caught: overclaim | Caught: all defects | Seconds per claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniCheck-Flan-T5-Large | 84 of 105 (80%) | 28 of 40 (70%) | 29 of 35 (83%) | 76 of 80 (95%) | 79 of 80 (99%) | 212 of 235 (90%) | 0.7 |
| HHEM-2.1-Open | 86 of 105 (82%) | 29 of 40 (72%) | 12 of 35 (34%) | 77 of 80 (96%) | 73 of 80 (91%) | 191 of 235 (81%) | 0.2 |
| Judge: Gemma 4 12B | 94 of 105 (90%) | 35 of 40 (88%) | 29 of 35 (83%) | 79 of 80 (99%) | 79 of 80 (99%) | 222 of 235 (94%) | 9.9 |
| MiniCheck or judge flags it | 78 of 105 (74%) | 37 of 40 (92%) | 33 of 35 (94%) | 79 of 80 (99%) | 80 of 80 (100%) | 229 of 235 (97%) | 10.6 |
| HHEM or judge flags it | 82 of 105 (78%) | 36 of 40 (90%) | 30 of 35 (86%) | 80 of 80 (100%) | 80 of 80 (100%) | 226 of 235 (96%) | 10.1 |

## How the report sorts claims

With `--judge-url` and the default checker, the report shows the judge's verdict and puts a claim under
"Check by hand" when the checker model disagrees. On this set:

| | Pass clean | Check by hand | Flagged |
| --- | --- | --- | --- |
| 105 good claims | 78 | 18 | 9 |
| 235 damaged claims | 6 | 17 | 212 |

So 6 of 235 defects (3%) got through with both models agreeing, and a reader reviews about one claim in
four by hand.

## What changed during the run

The first checker run removed only simple attributions ("Google's help page says …") before scoring.
MiniCheck then passed 75 of 105 originals (71%) and caught 225 of 252 defects; most of its false alarms
were claims framed as "North Carolina: the article says …", "Under the 2025 LSA terms, …" or
"counsel (Inside Track, Jan 2021) wrote that …". Widening the stripping raised originals passed to 80%
and cost some number catches, because a changed date inside stripped framing is no longer seen by the
checker (the judge still sees it). Those first scores are in `before-wider-stripping/`.

## Limits

- The originals were graded OK by two Claude models against their quotes, not against the sentences
  retrieved here. Some "false alarms" may be claims that say slightly more than the page.
- A swapped page from another website can happen to say the same thing; a few "missed swaps" may be
  correct passes.
- One judge model, one run, one machine. The defects are mechanical; real errors in research reports
  are subtler, so these rates are an upper bound on what to expect.
