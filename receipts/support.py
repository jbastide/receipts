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

def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class MiniCheck:
    """MiniCheck-Flan-T5-Large (Tang et al. 2024, MIT licence). Input and scoring follow its reference code."""
    name = "minicheck-flan-t5-large"
    repo, revision = "lytang/MiniCheck-Flan-T5-Large", "main"

    def __init__(self, batch_size: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.batch_size, self.device = torch, batch_size, _device()
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

    def __init__(self, batch_size: int = 8):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification
        self.torch, self.batch_size, self.device = torch, batch_size, _device()
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
        body = {"model": self.model or "default", "temperature": temperature, "max_tokens": 400,
                "messages": [{"role": "system", "content": JUDGE_SYSTEM},
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


class SupportChecker:
    def __init__(self, checker=None, judge: Judge | None = None, threshold: float = 0.5):
        if checker is None and judge is None:
            raise ValueError("support needs a checker, a judge, or both")
        self.checker, self.judge, self.threshold = checker, judge, threshold

    def assess(self, claim: str, quotes: list[str], url: str, page_text: str) -> Support:
        sents = page_sentences(page_text)
        if not claim.strip() or not sents:
            return Support(url, "not_checked", reason="no claim text" if sents else "no page text")
        ranked = rank_sentences(sents, claim + " " + " ".join(quotes), k=8)
        if not ranked:
            return Support(url, "not_supported", checker_score=None, reason="no sentence on the page shares a word with the claim")

        score, best_window = None, None
        if self.checker is not None:
            wins = windows(sents, ranked[:3])
            bare = strip_attribution(claim)
            scores = self.checker.score([(" ".join(sents[a:b]), bare) for a, b in wins])
            k = max(range(len(scores)), key=scores.__getitem__)
            score, best_window = round(scores[k], 3), wins[k]

        answer = None
        if self.judge is not None:
            idx = context(sents, ranked)
            answer = self.judge.judge(claim, [(i + 1, sents[i]) for i in idx])

        if answer is not None:
            verdict = answer.verdict
            evidence = [sents[n - 1] for n in answer.evidence]
            reason = answer.reason
        elif self.judge is not None and self.checker is None:
            return Support(url, "not_checked", reason="the judge gave no usable answer")
        else:
            verdict = "supported" if score >= self.threshold else "not_supported"
            a, b = best_window
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
        return Support(url, verdict, score, answer.verdict if answer else "", evidence, reason, flag)
