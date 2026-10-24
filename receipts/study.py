"""Code study items against a codebook with local models, measure each model, and compare two coders.

A study is a TOML file:

    name = "..."
    unit_label = "thread"            # what an item is, for the prompt
    instructions = "..."             # how to apply the codebook
    items = "items.jsonl"            # from a collector; paths are relative to the TOML file
    gold = "gold.csv"                # id plus one true/false column per field
    calibration_ids = "calibration_ids.txt"   # gold items used to accept or reject a model per field
    min_kappa = 0.6
    chunk_words = 2500

    [[fields]]
    name = "NO_LLM_DATA"
    definition = "..."

    [coders.gemma]
    url = "http://127.0.0.1:8197/v1"
    model = "gemma-4-12b-it-qat-q4_0"

The coder never writes text from the item: for every field that applies it names the unit (post,
comment, sentence) that shows it, by number. Long items are cut into chunks of whole units, each with
the first unit (the post) for context; a field counts when any chunk shows it.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import tomllib
from dataclasses import dataclass, field

import httpx


@dataclass
class Study:
    path: str
    name: str
    unit_label: str
    instructions: str
    fields: list[dict]
    items_path: str
    gold_path: str
    calibration_ids_path: str
    min_kappa: float = 0.6
    chunk_words: int = 2500
    coders: dict = field(default_factory=dict)
    gate: str = ""                 # a field that must apply before the other fields can
    ungated: list = field(default_factory=list)   # fields that may apply without the gate
    verify_instructions: str = ""  # the second, narrow question asked about each proposed code

    @property
    def out(self) -> str:
        d = os.path.join(os.path.dirname(self.path), "out")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def names(self) -> list[str]:
        return [f["name"] for f in self.fields]


def load_study(path: str) -> Study:
    with open(path, "rb") as fh:
        t = tomllib.load(fh)
    here = os.path.dirname(os.path.abspath(path))
    rel = lambda p: os.path.join(here, p) if p else ""
    return Study(os.path.abspath(path), t["name"], t.get("unit_label", "item"), t.get("instructions", ""),
                 t["fields"], rel(t.get("items", "items.jsonl")), rel(t.get("gold", "")),
                 rel(t.get("calibration_ids", "")), float(t.get("min_kappa", 0.6)),
                 int(t.get("chunk_words", 2500)), t.get("coders", {}), t.get("gate", ""), t.get("ungated", []), t.get("verify_instructions", ""))


def read_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_items(items: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)


# ---- chunking and the coding prompt ------------------------------------------------------------

def chunks(units: list[dict], max_words: int) -> list[list[int]]:
    """Unit indices per chunk: the first unit in every chunk, then whole units up to `max_words`."""
    if not units:
        return []
    head = len(units[0]["text"].split())
    out, cur, n = [], [0], head
    for i in range(1, len(units)):
        w = min(len(units[i]["text"].split()), max_words)
        if len(cur) > 1 and n + w > max_words:
            out.append(cur)
            cur, n = [0], head
        cur.append(i)
        n += w
    out.append(cur)
    return out


def system_prompt(s: Study) -> str:
    defs = "\n".join(f"{f['name']}: {' '.join(f['definition'].split())}" for f in s.fields)
    return (f"{s.instructions.strip()}\n\nFIELDS (use these exact names)\n{defs}\n\n"
            f"The {s.unit_label} is split into numbered units. For every field that applies, give its name "
            "and the number of the unit that best shows it; you may list a field twice with two units. "
            "Leave out fields that do not apply. If none apply, return an empty list.\n\n"
            'Answer in JSON: {"codes": [{"field": "NAME", "unit": NUMBER}]}')


def schema(s: Study) -> dict:
    return {"type": "object", "required": ["codes"], "properties": {"codes": {
        "type": "array", "maxItems": 2 * len(s.fields),
        "items": {"type": "object", "required": ["field", "unit"],
                  "properties": {"field": {"type": "string", "enum": s.names}, "unit": {"type": "integer"}}}}}}


def user_prompt(s: Study, item: dict, idx: list[int], max_words: int) -> str:
    meta = " | ".join(f"{k}: {v}" for k, v in item.get("meta", {}).items() if v not in (None, ""))
    lines = []
    for n, i in enumerate(idx, 1):
        ws = item["units"][i]["text"].split()
        text = " ".join(ws[:max_words]) + (" …" if len(ws) > max_words else "")
        lines.append(f"[{n}] ({item['units'][i]['id']}) {text}")
    return f"{s.unit_label.upper()} {item['id']} | {meta}\n\n" + "\n".join(lines)


class Coder:
    def __init__(self, name: str, url: str, model: str = "", timeout: float = 600.0, client: httpx.Client | None = None):
        self.name, self.model = name, model
        self.url = url.rstrip("/") + "/chat/completions"
        self.client = client or httpx.Client(timeout=timeout)

    def ask(self, system: str, user: str, sch: dict, max_tokens: int = 600) -> dict | None:
        """One JSON answer that has every key the schema requires, or None after a retry."""
        body = {"model": self.model or "default", "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "codes", "strict": True, "schema": sch}},
                "chat_template_kwargs": {"enable_thinking": False}}
        for _ in range(2):
            try:
                r = self.client.post(self.url, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", text, re.S)
                d = json.loads(m.group(0) if m else text)
                if isinstance(d, dict) and all(k in d for k in sch.get("required", [])):
                    if "codes" not in sch.get("required", []) or isinstance(d["codes"], list):
                        return d
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue
        return None


def code_item(s: Study, coder: Coder, item: dict) -> dict:
    """Values and evidence for one item. A code that cites a unit not shown is dropped and counted."""
    system, sch = system_prompt(s), schema(s)
    values = {f: False for f in s.names}
    evidence = {f: [] for f in s.names}
    dropped, failed = 0, 0
    parts = chunks(item["units"], s.chunk_words)
    for idx in parts:
        a = coder.ask(system, user_prompt(s, item, idx, s.chunk_words), sch)
        if a is None:
            failed += 1
            continue
        for c in a["codes"]:
            f, u = c.get("field"), c.get("unit")
            if f not in values or not isinstance(u, int) or not 1 <= u <= len(idx):
                dropped += 1
                continue
            values[f] = True
            uid = item["units"][idx[u - 1]]["id"]
            if uid not in evidence[f]:
                evidence[f].append(uid)
    apply_gate(s, values, evidence)
    return {"id": item["id"], "coder": coder.name, "model": coder.model, "values": values,
            "evidence": evidence, "chunks": len(parts), "failed_chunks": failed, "dropped_codes": dropped}


def apply_gate(s: Study, values: dict, evidence: dict):
    if s.gate and not values.get(s.gate, True):          # e.g. an off-topic thread gets no categories
        for f in values:
            if f != s.gate and f not in s.ungated and values[f]:
                values[f], evidence[f] = False, []


# ---- verifying proposed codes ------------------------------------------------------------------

VERIFY_SCHEMA = {"type": "object", "required": ["applies"], "properties": {"applies": {"type": "boolean"}}}


def verify_prompt(s: Study, f: dict) -> str:
    return (f"{s.verify_instructions.strip()}\n\nFIELD {f['name']}: {' '.join(f['definition'].split())}\n\n"
            'Answer in JSON: {"applies": true} or {"applies": false}')


def verify_item(s: Study, coder: Coder, item: dict, row: dict, context_words: int = 300, unit_words: int = 600) -> dict:
    """Ask one yes/no question per proposed code, showing the post and the units the code cited."""
    units = {u["id"]: u["text"] for u in item["units"]}
    head = " ".join(item["units"][0]["text"].split()[:context_words]) if item["units"] else ""
    values, evidence = dict(row["values"]), {k: list(v) for k, v in row["evidence"].items()}
    asked = rejected = 0
    for f in s.fields:
        name = f["name"]
        if not values.get(name):
            continue
        cited = [u for u in evidence.get(name, []) if u != "post"][:2]
        body = "\n".join(f"({u}) {' '.join(units.get(u, '').split()[:unit_words])}" for u in cited)
        user = f"{s.unit_label.upper()} {item['id']}\n\n(post) {head}" + (f"\n\n{body}" if body else "")
        a = coder.ask(verify_prompt(s, f), user, VERIFY_SCHEMA, max_tokens=20)
        asked += 1
        if a is not None and a.get("applies") is False:
            values[name], evidence[name] = False, []
            rejected += 1
    apply_gate(s, values, evidence)
    return {**row, "coder": f"{row['coder']}+verify", "proposed": row["values"], "values": values,
            "evidence": evidence, "verify_questions": asked, "verify_rejected": rejected}


def verify_study(s: Study, coder: Coder, source: str, progress=None) -> str:
    """Verify every item in codes-SOURCE.jsonl; writes codes-SOURCE-verified.jsonl (resumable)."""
    rows = read_codes(os.path.join(s.out, f"codes-{source}.jsonl"))
    path = os.path.join(s.out, f"codes-{source}-verified.jsonl")
    done = set(read_codes(path)) if os.path.exists(path) else set()
    items = {i["id"]: i for i in read_items(s.items_path)}
    with open(path, "a", encoding="utf-8") as fh:
        for n, (iid, row) in enumerate(rows.items(), 1):
            if iid in done:
                continue
            t = time.time()
            out = verify_item(s, coder, items[iid], row)
            out["verify_seconds"] = round(time.time() - t, 1)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                progress(n, len(rows))
    return path


def code_study(s: Study, coder: Coder, limit: int = 0, progress=None, ids: list[str] | None = None,
               tag: str = "") -> str:
    """Code every item, or only `ids` (resumable: items already in the output file are skipped).
    `tag` names a separate output file, e.g. for a prompt being tried on the calibration items."""
    path = os.path.join(s.out, f"codes-{coder.name}{'-' + tag if tag else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    items = read_items(s.items_path)
    if ids is not None:
        keep = set(ids)
        items = [i for i in items if i["id"] in keep]
    if limit:
        items = items[:limit]
    with open(path, "a", encoding="utf-8") as fh:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            t = time.time()
            row = code_item(s, coder, item)
            row["seconds"] = round(time.time() - t, 1)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                progress(n, len(items))
    return path


# ---- measuring --------------------------------------------------------------------------------------

def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary codings; None when undefined (both constant and equal)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None
    return round((po - pe) / (1 - pe), 3)


def read_codes(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(l) for l in fh if l.strip())}


def read_gold(path: str, names: list[str]) -> dict[str, dict[str, bool]]:
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: {f: r[f].strip().lower() == "true" for f in names} for r in csv.DictReader(fh)}


def score(names, ids, a: dict, b: dict, a_key=lambda r, f: r["values"][f], b_key=lambda r, f: r[f]) -> list[dict]:
    rows = []
    for f in names:
        x = [a_key(a[i], f) for i in ids]
        y = [b_key(b[i], f) for i in ids]
        rows.append({"field": f, "n": len(ids), "a_true": sum(x), "b_true": sum(y),
                     "agree_pct": round(100 * sum(p == q for p, q in zip(x, y)) / len(ids), 1) if ids else None,
                     "kappa": kappa(x, y)})
    return rows


def calibrate(s: Study, coder_name: str) -> tuple[list[dict], list[dict]]:
    """Per field, the coder against gold on the calibration items (decides pass) and on the rest (reported)."""
    codes = read_codes(os.path.join(s.out, f"codes-{coder_name}.jsonl"))
    gold = read_gold(s.gold_path, s.names)
    cal = [l.strip() for l in open(s.calibration_ids_path, encoding="utf-8") if l.strip()]
    cal = [i for i in cal if i in codes and i in gold]
    held = [i for i in gold if i in codes and i not in set(cal)]
    on_cal = score(s.names, cal, codes, gold)
    for r in on_cal:
        r["passes"] = r["kappa"] is not None and r["kappa"] >= s.min_kappa
    on_held = score(s.names, held, codes, gold)
    _write_csv(os.path.join(s.out, f"calibration-{coder_name}.csv"), on_cal)
    _write_csv(os.path.join(s.out, f"heldout-{coder_name}.csv"), on_held)
    return on_cal, on_held


def compare(s: Study, a_name: str, b_name: str) -> list[dict]:
    """Agreement between two coders; disagreements go to review.csv with each coder's cited units."""
    a = read_codes(os.path.join(s.out, f"codes-{a_name}.jsonl"))
    b = read_codes(os.path.join(s.out, f"codes-{b_name}.jsonl"))
    ids = [i for i in a if i in b]
    rows = score(s.names, ids, a, b, b_key=lambda r, f: r["values"][f])
    items = {i["id"]: i for i in read_items(s.items_path)}
    review = []
    for i in ids:
        units = {u["id"]: u["text"] for u in items.get(i, {}).get("units", [])}
        for f in s.names:
            if a[i]["values"][f] != b[i]["values"][f]:
                yes = a if a[i]["values"][f] else b
                cited = yes[i]["evidence"][f][:2]
                review.append({"id": i, "field": f, a_name: a[i]["values"][f], b_name: b[i]["values"][f],
                               "cited_by": yes[i]["coder"], "units": " | ".join(cited),
                               "text": " | ".join(" ".join(units.get(u, "").split())[:400] for u in cited)})
    _write_csv(os.path.join(s.out, f"compare-{a_name}-{b_name}.csv"), rows)
    _write_csv(os.path.join(s.out, f"review-{a_name}-{b_name}.csv"), review)
    return rows


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
