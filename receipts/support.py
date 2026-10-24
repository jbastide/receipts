"""Does the cited page support the claim?

Three parts, each replaceable:
  retrieval   picks the page sentences most likely to bear on the claim (BM25, no model)
  checker     a small model that scores premise -> claim support from 0 to 1 (MiniCheck or HHEM)
  judge       a local chat model behind an OpenAI-compatible endpoint (llama.cpp, Ollama, vLLM,
              LM Studio) that reads numbered page sentences and answers supported / partly /
              not supported, naming the sentences it relied on

The judge never writes a quotation: it returns sentence numbers, and the report shows those
sentences as they appear on the page.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .text import words

VERDICTS = ("supported", "partly", "not_supported")
_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it
its this that these those which who whom what when where how than then there their they them he she his
her we our you your i not no can may might must will would should could do does did has have had into
about over under also such so very more most other some any each per via""".split())


# ---- page sentences and retrieval ----------------------------------------------------------

def page_sentences(text: str, max_words: int = 80) -> list[str]:
    """Split page text into sentences; long runs without punctuation are cut every `max_words` words."""
    blocks = re.split(r"\n\s*\n|\s{3,}", text)
    out = []
    for b in blocks:
        for s in re.split(r"(?<=[.!?])[\"”’)]*\s+(?=[A-Z0-9\"“‘(])", b):
            ws = s.split()
            for i in range(0, len(ws), max_words):
                piece = " ".join(ws[i:i + max_words])
                if len(piece.split()) >= 3:
                    out.append(piece)
    return out


_PREFIX = r"(?:[^,.;:()]|\([^)]*\)){1,160}?"          # words, and parentheses that may hold commas
_ATTRIBUTION = re.compile(
    r"^(?:(?:according to|per|as stated (?:by|in|on))\s[^,]{1,120},\s*"
    r"|under (?:the )?[^,]{1,100},\s*"
    r"|[A-Z][\w .'&-]{1,40}:\s+"
    r"|" + _PREFIX + r"\b(?:says?|said|states?|stated|notes?|noted|reports?|reported|writes?|wrote|"
    r"explains?|explained|finds|found|shows?|showed|lists?|listed|confirms?|confirmed|warns?|warned|"
    r"argues?|argued|describes?|described|indicates?|indicated|adds?|added|places?|placed)\s+(?:that\s+)?)", re.I)


def strip_attribution(claim: str) -> str:
    """'North Carolina: the article says the lawyer must…' -> 'The lawyer must…'.

    The checker models read framing such as "Google's help page says", "Under the 2025 terms," or
    "North Carolina:" as facts to verify, and mark the claim down when the page doesn't state them.
    The judge is told source details need not appear and sees the full claim.
    """
    rest = claim.strip()
    for _ in range(2):                                   # framing can be nested: "X: the article says …"
        cut = _ATTRIBUTION.sub("", rest, count=1)
        if cut == rest or len(cut.split()) < 3:
            break
        rest = cut
    return rest[:1].upper() + rest[1:]


def _terms(s: str) -> list[str]:
    return [w for w in words(s) if w not in _STOP and len(w) > 1]


def rank_sentences(sents: list[str], query: str, k: int = 8) -> list[int]:
    """Indices of the `k` sentences that best match `query` (BM25), best first."""
    docs = [_terms(s) for s in sents]
    q = set(_terms(query))
    if not q or not docs:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1
    df = Counter(t for d in docs for t in set(d) if t in q)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(t for t in d if t in q)
        sc = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            sc += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append((sc, i))
    return [i for sc, i in sorted(scores, key=lambda x: (-x[0], x[1]))[:k] if sc > 0]


def windows(sents: list[str], centres: list[int], max_words: int = 400) -> list[tuple[int, int]]:
    """Stretches of consecutive sentences around each centre, up to `max_words`, without repeats."""
    out = []
    for c in centres:
        if any(a <= c < b for a, b in out):
            continue
        a, b, n = c, c + 1, len(sents[c].split())
        while True:
            grew = False
            if a > 0 and n + len(sents[a - 1].split()) <= max_words:
                a -= 1; n += len(sents[a].split()); grew = True
            if b < len(sents) and n + len(sents[b].split()) <= max_words:
                n += len(sents[b].split()); b += 1; grew = True
            if not grew:
                break
        out.append((a, b))
    return out


def context(sents: list[str], ranked: list[int], max_words: int = 900) -> list[int]:
    """The ranked sentences plus one neighbour each side, in page order, within `max_words`."""
    keep, n = set(), 0
    for i in ranked:
        for j in (i, i - 1, i + 1):
            if 0 <= j < len(sents) and j not in keep and n + len(sents[j].split()) <= max_words:
                keep.add(j)
                n += len(sents[j].split())
    return sorted(keep)


# ---- checkers --------------------------------------------------------------------------------

def _device(choice: str = "auto"):
    """'auto' picks CUDA, then Apple's GPU, then the CPU. On Apple silicon the checker and a llama.cpp
    judge sharing the GPU has crashed PyTorch, so the CLI passes 'cpu' when a judge is in use."""
    import torch
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class MiniCheck:
    """MiniCheck-Flan-T5-Large (Tang et al. 2024, MIT licence). Input and scoring follow its reference code."""
    name = "minicheck-flan-t5-large"
    repo, revision = "lytang/MiniCheck-Flan-T5-Large", "main"

    def __init__(self, batch_size: int = 4, device: str = "auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.batch_size, self.device = torch, batch_size, _device(device)
        self.tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.repo, revision=self.revision).to(self.device).eval()

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch, out = self.torch, []
        for i in range(0, len(pairs), self.batch_size):
            batch = ["predict: " + p + self.tok.eos_token + c for p, c in pairs[i:i + self.batch_size]]
            enc = self.tok(batch, max_length=2048, truncation=True, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
                logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0, :]
                probs = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1]   # 3: no support, 209: support
            out += probs.cpu().tolist()
        return out


class HHEM:
    """Vectara HHEM-2.1-Open (Apache-2.0). Weights loaded directly, so none of the repository's code is run."""
    name = "hhem-2.1-open"
    repo, revision = "vectara/hallucination_evaluation_model", "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"
    base = "google/flan-t5-base"
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {p}\n\nHypothesis: {c}"

    def __init__(self, batch_size: int = 8, device: str = "auto"):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification
        self.torch, self.batch_size, self.device = torch, batch_size, _device(device)
        cfg = AutoConfig.from_pretrained(self.base)
        cfg.num_labels = 2
        model = T5ForTokenClassification(cfg)
        state = load_file(hf_hub_download(self.repo, "model.safetensors", revision=self.revision))
        state = {k.removeprefix("t5."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if "embed_tokens" not in k] + list(unexpected)
        if bad:
            raise RuntimeError(f"HHEM weights did not load cleanly: {bad[:5]}")
        self.model = model.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.base)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch, out = self.torch, []
        for i in range(0, len(pairs), self.batch_size):
            batch = [self.prompt.format(p=p, c=c) for p, c in pairs[i:i + self.batch_size]]
            enc = self.tok(batch, max_length=2048, truncation=True, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, 0, :]
            out += torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        return out


CHECKERS = {"minicheck": MiniCheck, "hhem": HHEM}


# ---- judge -------------------------------------------------------------------------------------

JUDGE_SYSTEM = """You check whether a web page supports a claim. You are shown numbered sentences taken from the page. Judge only from those sentences, never from what you know.

supported: the sentences show the main thing the claim asserts. Details about the source (who wrote it, on which site or page, when) do not need to appear. A fair paraphrase or summary is supported.
partly: the sentences show part of the claim, but the claim also states another specific fact of substance that they do not show: a number, a date, a condition, an exception, a second rule, or listed items.
not_supported: the sentences do not show the claim's main assertion, or they contradict it.

Answer in JSON with three fields: "reason" (one short sentence), "evidence" (the numbers of up to five sentences that show what the claim asserts; empty if none), and "verdict" (supported, partly or not_supported)."""

# Used instead of the first two sentences of JUDGE_SYSTEM when a claim cites several pages; the
# single-page wording is kept exactly as benchmarked in bench/results/planted.
JUDGE_MULTI = ("You check whether the pages a claim cites support it. You are shown numbered sentences "
               "taken from those pages; each sentence says which website it is from, and the claim may "
               "draw different facts from different pages.")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "integer"}},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reason", "evidence", "verdict"],
}


@dataclass
class JudgeAnswer:
    verdict: str
    evidence: list[int]
    reason: str
    votes: list[str] = field(default_factory=list)


class Judge:
    def __init__(self, base_url: str, model: str = "", samples: int = 1, timeout: float = 300.0,
                 api_key: str = "", client: httpx.Client | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.samples = model, max(1, samples)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)
        self.name = f"judge:{model or base_url}"

    def _ask(self, claim: str, numbered: list[tuple[int, str]], temperature: float) -> JudgeAnswer | None:
        page = "\n".join(f"[{n}] {s}" for n, s in numbered)
        system = JUDGE_SYSTEM
        if any(s.startswith("(from ") for _, s in numbered):
            first = "You check whether a web page supports a claim. You are shown numbered sentences taken from the page."
            system = JUDGE_MULTI + JUDGE_SYSTEM[len(first):]
        body = {"model": self.model or "default", "temperature": temperature, "max_tokens": 400,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"CLAIM:\n{claim}\n\nPAGE SENTENCES:\n{page}"}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "strict": True, "schema": JUDGE_SCHEMA}},
                "chat_template_kwargs": {"enable_thinking": False}}
        for _ in range(2):                                      # one retry on a malformed answer
            try:
                r = self.client.post(self.url, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", text, re.S)
                a = json.loads(m.group(0) if m else text)
                verdict = str(a.get("verdict", "")).strip().lower().replace(" ", "_")
                if verdict not in VERDICTS:
                    continue
                valid = {n for n, _ in numbered}
                ev = [int(x) for x in a.get("evidence", []) if str(x).lstrip("-").isdigit() and int(x) in valid][:5]
                return JudgeAnswer(verdict, ev, str(a.get("reason", ""))[:300])
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue
        return None

    def decompose(self, claim: str) -> list[str] | None:
        body = {"model": self.model or "default", "temperature": 0.0, "max_tokens": 700,
                "messages": [{"role": "system", "content": SPLIT_SYSTEM},
                             {"role": "user", "content": f"CLAIM:\n{claim}"}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "facts", "strict": True, "schema": SPLIT_SCHEMA}},
                "chat_template_kwargs": {"enable_thinking": False}}
        for _ in range(2):
            try:
                r = self.client.post(self.url, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", text, re.S)
                facts = json.loads(m.group(0) if m else text).get("facts", [])
                facts = [str(f).strip() for f in facts if str(f).strip()][:8]
                if facts:
                    return facts
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue
        return None

    def judge(self, claim: str, numbered: list[tuple[int, str]]) -> JudgeAnswer | None:
        if self.samples == 1:
            return self._ask(claim, numbered, 0.0)
        answers = [a for a in (self._ask(claim, numbered, 0.7) for _ in range(self.samples)) if a]
        if not answers:
            return None
        votes = [a.verdict for a in answers]
        top, n = Counter(votes).most_common(1)[0]
        pick = next(a for a in answers if a.verdict == top)
        pick.votes = votes
        if n <= len(votes) / 2:                                 # no majority: a split vote
            pick.reason = f"split vote ({', '.join(votes)}): " + pick.reason
        return pick


# ---- splitting dense claims ------------------------------------------------------------------------

SPLIT_SYSTEM = """You split a claim from a research report into the separate facts it states, so that each fact can be checked against the sources the claim cites.

Rules:
- One fact per item, written as a short sentence that stands on its own: repeat the subject instead of writing "it" or "they".
- Use only words and figures that are in the claim. Add nothing, and keep every number, date and qualifier with the fact it belongs to.
- Keep a list together: "the model parses intent, budget and dislikes" is one fact, not three.
- Leave out the report's own framing that no source would state: its stage, section or row names, what it calls things, and its labels, verdicts and ratings such as evidence strength, risk level or a status column ("Documented", "Inferred", "Measured", "High", "Low"). Leave out the names of the sources cited.
- If the claim states only one fact, return it as the only item.
- Usually two to five items; never more than eight.

Answer in JSON: {"facts": ["...", "..."]}"""

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8}},
    "required": ["facts"],
}

DENSE_WORDS = 35        # a claim this long, with three figures or more, or from a table row, is split
DENSE_NUMBERS = 3
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
_FIGURE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(s: str) -> set[str]:
    """The figures in `s`, without thousands separators; arXiv ids are not figures."""
    s = _ARXIV.sub(" ", s)
    return {m.group().replace(",", "") for m in _FIGURE.finditer(s)}


def is_dense(claim: str) -> bool:
    return " | " in claim or len(claim.split()) >= DENSE_WORDS or len(numbers(claim)) >= DENSE_NUMBERS


def fact_from_claim(fact: str, claim: str, min_share: float = 0.8) -> bool:
    """True when a split-off fact adds nothing: its figures all come from the claim, and nearly all its words."""
    terms = {_stem(t) for t in _terms(fact)}
    if not terms or not numbers(fact) <= numbers(claim):
        return False
    return len(terms & {_stem(t) for t in _terms(claim)}) / len(terms) >= min_share


def _stem(w: str) -> str:
    """Crude plural folding, enough that 'drivers' matches 'driver' and 'brands' matches 'brand'."""
    for end in ("ies", "es", "s"):
        if len(w) > len(end) + 2 and w.endswith(end):
            return w[: -len(end)] + ("y" if end == "ies" else "")
    return w


def split_claim(claim: str, judge: "Judge") -> tuple[list[str], list[str]]:
    """Facts a dense claim states, checked against the claim, and the claim's figures no fact covers."""
    facts = [f for f in (judge.decompose(claim) or []) if fact_from_claim(f, claim)]
    facts = list(dict.fromkeys(facts))
    if len(facts) < 2:
        return [], []
    covered = set().union(*(numbers(f) for f in facts))
    body = claim.split(" | ", 1)[1] if " | " in claim else claim     # a table row's label cell holds no figures to check
    left_out = sorted(numbers(body) - covered, key=lambda n: claim.find(n))
    return facts, left_out


# ---- putting it together -------------------------------------------------------------------------

@dataclass
class Support:
    url: str
    verdict: str                    # supported, partly, not_supported, or not_checked
    checker_score: float | None = None
    judge_verdict: str = ""
    evidence: list[str] = field(default_factory=list)
    reason: str = ""
    flag: str = ""                  # why a person should look
    facts: list[dict] = field(default_factory=list)    # when a dense claim was split: one entry per fact
    left_out: list[str] = field(default_factory=list)  # figures in the claim that no split-off fact covers


class SupportChecker:
    def __init__(self, checker=None, judge: Judge | None = None, threshold: float = 0.5, split: bool = True):
        if checker is None and judge is None:
            raise ValueError("support needs a checker, a judge, or both")
        self.checker, self.judge, self.threshold = checker, judge, threshold
        self.split = split and judge is not None

    def assess(self, claim: str, quotes: list[str], url: str, page_text: str) -> Support:
        return self.assess_pages(claim, quotes, [(url, page_text)])

    def assess_pages(self, claim: str, quotes: list[str], pages: list[tuple[str, str]]) -> Support:
        """A dense claim is split into single facts, each judged on its own; otherwise judged whole."""
        if self.split and is_dense(claim):
            facts, left_out = split_claim(claim, self.judge)
            if facts:
                return self._combine([(f, self._assess_one(f, [], pages)) for f in facts], left_out, pages)
        return self._assess_one(claim, quotes, pages)

    def _combine(self, parts: list[tuple[str, Support]], left_out: list[str], pages) -> Support:
        checked = [(f, s) for f, s in parts if s.verdict != "not_checked"]
        verdicts = {s.verdict for _, s in checked}
        if not checked:
            verdict = "not_checked"
        elif verdicts == {"supported"}:
            verdict = "supported"
        elif verdicts == {"not_supported"}:
            verdict = "not_supported"
        else:
            verdict = "partly"
        n_ok = sum(1 for _, s in checked if s.verdict == "supported")
        weak = [f for f, s in checked if s.verdict != "supported"]
        reason = f"{n_ok} of {len(parts)} facts supported" + (f"; not shown: {weak[0]}" if weak else "")
        scores = [s.checker_score for _, s in parts if s.checker_score is not None]
        flags = [f"fact {i}: {s.flag}" for i, (_, s) in enumerate(parts, 1) if s.flag]
        if left_out:
            flags.insert(0, "figures in the claim that no checked fact covers: " + ", ".join(left_out))
        evidence = [s.evidence[0] for _, s in parts if s.verdict == "supported" and s.evidence][:4]
        first = next((s.url for _, s in parts if s.verdict == "supported"), parts[0][1].url)
        judged = {s.judge_verdict for _, s in parts if s.judge_verdict}
        return Support(first, verdict, min(scores) if scores else None,
                       verdict if judged else "", evidence, reason, "; ".join(flags),
                       facts=[{"text": f, "verdict": s.verdict, "checker_score": s.checker_score,
                               "judge_verdict": s.judge_verdict, "evidence": s.evidence, "reason": s.reason,
                               "url": s.url, "flag": s.flag} for f, s in parts],
                       left_out=left_out)

    def _assess_one(self, claim: str, quotes: list[str], pages: list[tuple[str, str]]) -> Support:
        """Judge a claim against all of its cited pages together.

        A claim that cites several sources usually draws one fact from each, so no single page backs
        all of it. The judge sees the best-matching sentences of every page, labelled by website.
        """
        first = pages[0][0] if pages else ""
        if not claim.strip():
            return Support(first, "not_checked", reason="no claim text")
        query = claim + " " + " ".join(quotes)
        found = []                                       # (url, sentences, ranked indices)
        for url, text in pages:
            sents = page_sentences(text)
            if sents:
                found.append((url, sents, rank_sentences(sents, query, k=8)))
        if not found:
            return Support(first, "not_checked", reason="no page text")
        if not any(r for _, _, r in found):
            return Support(first, "not_supported", reason="no sentence on the page shares a word with the claim")

        score, best, checker_error = None, None, ""
        if self.checker is not None:
            pairs, where = [], []
            bare = strip_attribution(claim)
            for url, sents, ranked in found:
                for a, b in windows(sents, ranked[:3]):
                    pairs.append((" ".join(sents[a:b]), bare))
                    where.append((url, sents, ranked, a, b))
            try:
                scores = self.checker.score(pairs)
            except Exception as e:                     # a device error on one claim must not end the run
                checker_error = f"the checker model failed ({type(e).__name__})"
                scores = []
            if scores:
                k = max(range(len(scores)), key=scores.__getitem__)
                score, best = round(scores[k], 3), where[k]

        answer, numbered, source_of = None, [], {}
        if self.judge is not None:
            budget = 900 if len(found) == 1 else max(400, 1500 // len(found))
            for url, sents, ranked in found:
                label = f"(from {urlparse(url).netloc}) " if len(found) > 1 else ""
                for i in context(sents, ranked, max_words=budget):
                    n = len(numbered) + 1
                    numbered.append((n, label + sents[i]))
                    source_of[n] = (url, sents[i])
            answer = self.judge.judge(claim, numbered)

        if answer is not None:
            verdict, reason = answer.verdict, answer.reason
            evidence = [source_of[n][1] for n in answer.evidence]
            url = source_of[answer.evidence[0]][0] if answer.evidence else (best[0] if best else first)
        elif best is None:
            return Support(first, "not_checked", reason=checker_error or "the judge gave no usable answer",
                           flag=checker_error)
        else:
            verdict = "supported" if score >= self.threshold else "not_supported"
            url, sents, ranked, a, b = best
            evidence = [sents[i] for i in ranked if a <= i < b][:2]
            reason = f"checker score {score}"

        flag = ""
        if answer is not None and score is not None:
            if verdict == "supported" and score < self.threshold:
                flag = "the judge says supported, the checker model does not"
            elif verdict == "not_supported" and score >= self.threshold:
                flag = "the checker model says supported, the judge does not"
        if answer is not None and "split vote" in answer.reason:
            flag = flag or "the judge's samples disagreed"
        if self.judge is not None and answer is None:
            flag = "the judge gave no usable answer; verdict from the checker model"
        if checker_error:
            flag = (flag + "; " if flag else "") + checker_error
        return Support(url, verdict, score, answer.verdict if answer else "", evidence, reason, flag)
