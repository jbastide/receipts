# receipts

Check an AI research report before you forward it or publish it. `receipts check` reads a report
from ChatGPT, Claude, Gemini, Perplexity or an open-source research agent, fetches every page it
cites, keeps a copy of each, and tells you:

- whether each quote is on the page it cites: **verbatim**, **elided** (has an ellipsis; every piece
  is on the page), **edited** (most of it is there, but trimmed, reworded or stitched), **missing**, or
  **unreadable** (we couldn't see the page's text);
- for edited and missing quotes, the nearest passage that *is* on the page;
- whether each link works, and for dead links, whether the Wayback Machine ever captured them, which
  separates a page that moved from one that may never have existed;
- with `--support`, whether the cited page supports each claim, quoted or not: **supported**,
  **partly** or **not supported**, with the page sentences the verdict rests on.

It runs on your machine. Nothing is sent anywhere except the requests to the cited pages and, for dead
links, to the Wayback Machine. The quote and link checks use no model; the support check uses small
models that run locally.

Working name. Steps 1 and 2 of 6 in the design draft (`original-data-tool/DESIGN.md` on the
`original-data-tool` branch of the tools repository).

## Install and run

```
uv sync
uv run receipts check report.md
```

Output goes to `report-check/`: `check.md` to read, `check.json` with everything, `check.csv` with
one row per quote. Fetched pages are kept in `~/.cache/receipts/pages` (change with `--cache`), so a
second run reads the same copies; `--offline` uses only those.

```
uv run receipts check runs/*.json -o checks/        # several reports at once
uv run receipts check report.md --fail-on missing,edited,gone   # exit 1 if any, for scripts
```

## What it reads

| Format | What it looks for |
| --- | --- |
| Markdown or plain text | Sentences with inline links, bare URLs, `[1]` or `[^1]` markers resolved through a Sources or References list or reference definitions; quotations in “curly” or "straight" quotes of five words or more; block quotes |
| Receipts JSON | `{"claims": [{"id", "claim", "url", "quote"}]}` or a bare list of such objects |
| OpenAI Responses JSON | `output_text` with `url_citation` annotations |
| Anthropic Messages JSON | text blocks with `citations`; `cited_text` is checked as the quote |
| Perplexity chat completion JSON | the message with its `citations` list |

The three API formats were built from the providers' documented response shapes and tested on
hand-made samples, not yet on real exports.

## How quotes are graded

The page is reduced to letters and digits (spacing, punctuation and case ignored) and the quote is
looked for in it. If it isn't there, the quote is cut into overlapping five-word runs: 80% or more
found is **edited**, less is **missing**. A page with fewer than 400 letters and digits of text is
**unreadable**: usually a bot block or a page built by JavaScript. These are the rules used in the
[Haiku, Sonnet or Opus study](https://tools.copyfororiginals.com/haiku-vs-sonnet-vs-opus-research-study/),
so its grades can be reproduced.

Reddit refuses plain requests, so Reddit threads are read from the
[Arctic Shift](https://arctic-shift.photon-reddit.com/) archive instead. Google help pages are
requested in English.

## Benchmark

On the nine published runs of the Haiku, Sonnet or Opus study (593 quotes), `receipts check` gives
the published grade for all 593. Its negative control matches the study's own: of 588 quotes of six
words or more, none is accepted as verbatim after one word is replaced, and 22 are accepted against a
different page (passages that appear on more than one of the cited pages).

```
uv run python bench/model_study.py path/to/haiku-vs-sonnet-vs-opus-research-study
```

Pages change, so a later run may find some quotes edited or missing.

## Claim support

A quote can be on the page word for word and still not back the claim made with it, and most claims
in a research report have no quote at all. `--support` checks each claim against the page it cites:

1. **Retrieval.** The page is split into sentences and the ones that best match the claim are picked
   (BM25, no model).
2. **Checker model.** A small model scores whether those passages support the claim, from 0 to 1:
   [MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) (770M, MIT) by
   default, or [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model)
   (flan-t5-base, Apache-2.0) with `--checker hhem`. Both run on a laptop CPU or GPU. HHEM's weights
   are loaded directly at a pinned revision, so none of its repository's code is run. A leading
   attribution ("Google's help page says …") is removed before scoring, because these models otherwise
   mark a claim down when the page doesn't name itself.
3. **Judge (optional).** A local chat model behind any OpenAI-compatible endpoint (llama.cpp,
   Ollama, vLLM, LM Studio) reads the numbered page sentences and answers supported, partly or not
   supported, naming the sentence numbers it relied on. It never writes a quotation; the report shows
   those sentences as they are on the page. Output is constrained to a JSON schema and validated;
   a malformed answer is retried once. `--judge-samples 3` takes a majority and flags split votes.

With both, the judge's verdict is reported and the claim is flagged for a person when the two
disagree.

```
uv sync --extra support                       # torch, transformers: about 2 GB
uv run receipts check report.md --support                         # checker model only
uv run receipts check report.md --judge-url http://127.0.0.1:8080/v1   # checker model + judge
uv run receipts check report.md --checker none --judge-url http://127.0.0.1:11434/v1 --judge-model gemma3:12b
```

The first run downloads the checker model from Hugging Face (MiniCheck about 3 GB, HHEM about 1 GB).

### How well it works

On 340 planted cases built from the model study's graded claims (`bench/results/planted/`), on a
16 GB MacBook Air:

| Method | Good claims passed | Damaged claims caught | Seconds per claim |
| --- | --- | --- | --- |
| MiniCheck alone | 80% | 90% | 0.7 |
| HHEM alone | 82% | 81% (misses most negations) | 0.2 |
| Gemma 4 12B judge alone | 90% | 94% | 9.9 |
| MiniCheck and the judge, as the report uses them | 74% clean, 17% check by hand | 90% flagged, 7% check by hand, 3% missed | 10.6 |

The checker model alone is fast enough for a whole report in seconds but flags one good claim in five.
Add a judge when you want the report to be usable as is.

On a real report the picture is rougher. A 112-claim research synthesis (prose and comparison tables,
most claims carrying several figures) came back, with MiniCheck and the Gemma judge, as 43 supported,
60 partly, 7 not supported and 2 not checked, in about 20 minutes. "Partly" there mostly means the claim
packs three to six specifics and the judge was shown sentences for only some of them. When a claim
cites several pages, the judge reads sentences from all of them together, labelled by website (that
moved 11 of 50 multi-source claims from partly to supported). Splitting dense claims into single facts
before judging is the next improvement.

## What it does not do

- Support verdicts are a reason to look, not a ruling. The models read the sentences that best match
  the claim; a claim drawn from another part of the page can be marked unsupported.
- Quoted phrases under five words are not graded as quotes.

## Tests

```
uv run pytest
```

The tests run offline against a fake web.

## Licence

Not chosen yet.
