# Re-coding the Reddit study with two local models (October 25, 2026)

The Reddit study (`ai-visibility-measurement-reddit-study/` in the tools repository) coded 308 threads
against a 12-category codebook with Claude: a Sonnet first pass and an Opus published pass, which
agreed at kappa 0.65–0.91 per category. Here the same 308 threads are re-coded on a 16 GB MacBook Air
with two local models, Gemma 4 12B (QAT, 4-bit) and Qwen3.5 9B (Q6_K), through `receipts study`.

**The "gold" is the published Opus pass, not human coding.** Kappa here measures agreement with that
pass.

## Protocol

- Threads from the Arctic Shift archive, one unit per post or comment (deleted and removed comments
  skipped); long threads cut into 2,500-word chunks that each repeat the post. 308 threads, 6,545
  comments, 380 chunks.
- The coder lists each field that applies with the number of the unit that shows it; a code citing a
  unit it was not shown is dropped. Off-topic threads get no categories (the codebook's rule).
- Optional verification: each proposed code gets one yes/no question showing the post and the cited
  comments.
- 60 threads (seed 7) for calibration, 248 held out. Prompts were tried only on the calibration
  threads: the first prompt found 97% of the gold codes at 40% precision; a stricter prompt and an
  explicit relevance gate lowered recall to 38% (kappa got worse); generous proposing plus
  verification was kept. The held-out threads were coded once, with the frozen prompt.
- 95% bootstrap intervals on every kappa (1,000 resamples).

## Results (held-out kappa against the published pass, n = 248)

See `table.md` and `kappa.csv`. For each field the setup (coder, with or without verification) is
chosen by calibration kappa only, and its held-out kappa is reported.

- **5 of 14 fields reach 0.6** with the chosen setup: healthcare 0.92, TOOL_COST_ACCURACY 0.76,
  FAN_OUT 0.66, FORMAT_CLAIMS 0.66, NONDETERMINISM 0.61. The others run 0.33 to 0.60. The published
  Sonnet–Opus agreement was 0.65 to 0.91.
- **Verification helps a generous coder and hurts a cautious one.** Gemma proposed 1,532 codes and
  verification rejected 506; it raised Gemma's held-out kappa on 14 of 14 fields. Qwen proposed 906
  and verification rejected 486, lowering most fields (TOOL_COST_ACCURACY 0.59 to 0.13).
- **The two local models agree with each other less than with the gold** (Gemma+verify vs Qwen+verify:
  kappa 0.18 to 0.58), so two-model agreement alone would not have found the good fields.
- **60 calibration items cannot settle a field.** Calibration intervals span roughly 0.3 to 0.9; two
  fields that passed on calibration with Gemma fell below 0.6 on the held-out threads, and two that
  failed came out above it.
- Qwen returned no usable answer for 19 of 380 chunks; Gemma for none.
- Time: about 2.5 hours per model for coding and 1 hour for verification.

## What this means for the tool

A 9–12B model on a laptop is a usable first pass for a codebook like this one, not a replacement for a
frontier coder or a person: the gate needs a few hundred labelled items to be trustworthy, the setup
should be chosen per field, and disagreements should go to review. Codes are in
`../../studies/reddit-aeo/out/`; the thread text is not committed (it is Reddit content and is
re-collected from the archive by `receipts study collect`).
