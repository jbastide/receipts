# receipts

Check an AI research report before you forward it or publish it. `receipts check` reads a report
from ChatGPT, Claude, Gemini, Perplexity or an open-source research agent, fetches every page it
cites, keeps a copy of each, and tells you:

- whether each quote is on the page it cites: **verbatim**, **elided** (has an ellipsis; every piece
  is on the page), **edited** (most of it is there, but trimmed, reworded or stitched), **missing**, or
  **unreadable** (we couldn't see the page's text);
- for edited and missing quotes, the nearest passage that *is* on the page;
- whether each link works, and for dead links, whether the Wayback Machine ever captured them, which
  separates a page that moved from one that may never have existed.

It runs on your machine. Nothing is sent anywhere except the requests to the cited pages and, for dead
links, to the Wayback Machine. No model is used at this step.

Working name. Step 1 of 6 in the design draft (`original-data-tool/DESIGN.md` on the
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

## What it does not do yet

It does not judge whether a page *supports* a claim. A quote can be on the page word for word and
still not back the claim made with it. That is step 2 (a small local checker model plus a judge).

## Tests

```
uv run pytest
```

The tests run offline against a fake web.

## Licence

Not chosen yet.
