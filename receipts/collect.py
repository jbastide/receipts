"""Collectors: turn a source into study items, each split into numbered units a coder can cite.

An item is one JSON line: {"id", "meta": {...}, "units": [{"id", "text"}]}.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

from .fetch import Fetcher

ARCTIC = "https://arctic-shift.photon-reddit.com/api/"
_SKIP = {"[deleted]", "[removed]", ""}


RETRY = {0, 422, 429, 500, 502, 503, 504}   # Arctic Shift answers 422 "Timeout. Maybe slow down a bit"


def _get(fetcher: Fetcher, path: str, params: dict, pause: float, attempts: int = 6) -> dict:
    """One Arctic Shift request, cached by the fetcher; waits `pause` seconds after a live request and
    retries a busy or failed answer with a growing wait, never trusting a stored failure."""
    url = ARCTIC + path + "?" + urlencode(params)
    f = fetcher.get(url)
    for n in range(attempts):
        if f.status == 200:
            break
        if f.status not in RETRY or n == attempts - 1:
            raise RuntimeError(f"Arctic Shift answered {f.status or f.error} for {path} {params}")
        time.sleep(pause * 10 * (n + 1))
        f = fetcher.get(url, refresh=True)
    if not f.from_cache:
        time.sleep(pause)
    return json.loads(f.body())


def _date(ts) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).date().isoformat()


def reddit_threads(ids: list[str], fetcher: Fetcher, pause: float = 1.0, progress=None) -> list[dict]:
    """Posts and every comment from the Arctic Shift archive, one item per thread, in `ids` order."""
    posts = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        d = _get(fetcher, "posts/ids", {"ids": ",".join(chunk),
                                        "fields": "id,subreddit,created_utc,title,selftext,num_comments"}, pause)
        posts.update({p["id"]: p for p in d.get("data") or []})
    items = []
    for n, pid in enumerate(ids, 1):
        p = posts.get(pid)
        if p is None:
            continue
        units = [{"id": "post", "text": (p.get("title") or "") + "\n" + (p.get("selftext") or "")}]
        seen, after = set(), 1104537600                         # 2005-01-01
        while True:
            d = _get(fetcher, "comments/search", {"link_id": pid, "sort": "asc", "after": after, "limit": 100,
                                                   "fields": "id,body,created_utc"}, pause)
            data = d.get("data") or []
            new = [c for c in data if c["id"] not in seen]
            for c in new:
                seen.add(c["id"])
                if (c.get("body") or "").strip() not in _SKIP:
                    units.append({"id": f"c:{c['id']}", "text": c["body"]})
            if len(data) < 100 or not new:
                break
            after = max(c["created_utc"] for c in data) - 1
        items.append({"id": pid, "meta": {"subreddit": p.get("subreddit"), "date": _date(p["created_utc"]),
                                          "title": p.get("title"), "num_comments": p.get("num_comments")},
                      "units": units})
        if progress:
            progress(n, len(ids))
    return items
