"""Build this study's files from the published Reddit study in the tools repository.

Usage: python3 make.py ~/dev/tools/ai-visibility-measurement-reddit-study
Writes study.toml (fields copied from method/CODEBOOK.txt), gold.csv (the published coding, from
threads.csv), ids.txt (the 308 thread ids) and calibration_ids.txt (60 ids, seed 7).
The gold labels are the published pass (Claude Opus), not human coding.
"""
import csv, os, random, re, sys

src = os.path.expanduser(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
book = open(os.path.join(src, "method", "CODEBOOK.txt"), encoding="utf-8").read()
cats = book.split("CATEGORIES (use these exact codes)")[1].split("Also set:")[0]
fields = []
for m in re.finditer(r"^([A-Z0-9_]{4,})\s+(.*?)(?=^[A-Z0-9_]{4,}\s|\Z)", cats, re.M | re.S):
    d = " ".join(m.group(2).split())
    d = re.sub(r"\s*Put the claim's direction in the quote choice[^.]*\.", "", d)   # about quoting, not coding
    fields.append((m.group(1), d))
also = book.split("Also set:")[1]
for name in ("relevant", "healthcare"):
    m = re.search(rf"^{name}:\s+(.*?)(?=^\w+:|\Z)", also, re.M | re.S)
    fields.append((name, " ".join(m.group(1).split()).replace("true if", "Applies if")))

# The proposer is generous on purpose (in calibration it found 97% of the published codes); each
# proposed code is then checked with one narrow yes/no question (verify_instructions).
intro = ("You are coding Reddit threads (a post and its comments) about AI search and SEO. Decide which "
         "fields apply anywhere in the thread, in the post or any comment. A field applies when someone "
         "describes or discusses that problem or claim; a passing keyword is not enough. A thread can have "
         "no fields, one or many. 'relevant' applies if the thread discusses measuring, tracking, "
         "reporting or attributing AI search or AI Overview visibility or traffic at all.")
verify = ("You check one code given to a Reddit thread about AI search and SEO. You see the start of the post "
          "and the comments the code was based on. Answer true only if someone in this text clearly raises "
          "the specific problem or claim the field describes, as more than a passing remark or a keyword. "
          "Answer false if it is only mentioned in passing, if it is a different problem, or if you are unsure.")
q = lambda s: '"""' + s.replace('"""', "'''") + '"""'
toml = [f'name = "AI visibility measurement threads (Reddit study re-coded)"', 'unit_label = "thread"',
        f"instructions = {q(intro)}", f"verify_instructions = {q(verify)}", 'items = "items.jsonl"', 'gold = "gold.csv"',
        'calibration_ids = "calibration_ids.txt"', "min_kappa = 0.6", "chunk_words = 2500",
        'gate = "relevant"', 'ungated = ["healthcare"]', ""]
for name, d in fields:
    toml += ["[[fields]]", f'name = "{name}"', f"definition = {q(d)}", ""]
toml += ["[coders.gemma]", 'url = "http://127.0.0.1:8197/v1"', 'model = "gemma-4-12b-it-qat-q4_0"', "",
         "[coders.qwen]", 'url = "http://127.0.0.1:8197/v1"', 'model = "qwen3.5-9b"', ""]
open(os.path.join(here, "study.toml"), "w", encoding="utf-8").write("\n".join(toml))

rows = list(csv.DictReader(open(os.path.join(src, "threads.csv"), encoding="utf-8")))
names = [n for n, _ in fields]
with open(os.path.join(here, "gold.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id"] + names)
    for r in rows:
        cats_ = set(filter(None, r["categories"].split(";")))
        w.writerow([r["id"]] + [str(n in cats_).lower() for n in names[:-2]] +
                   [r["relevant"].lower(), r["healthcare"].lower()])
ids = [r["id"] for r in rows]
open(os.path.join(here, "ids.txt"), "w").write("\n".join(ids) + "\n")
cal = sorted(random.Random(7).sample(ids, 60))
open(os.path.join(here, "calibration_ids.txt"), "w").write("\n".join(cal) + "\n")
print(len(fields), "fields:", ", ".join(names)); print(len(ids), "threads;", len(cal), "for calibration")
