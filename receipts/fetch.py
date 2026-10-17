"""Fetch a URL once, keep the body, and record what came back.

Every fetch is stored under the cache directory as `<key>.body` (the bytes as received) and
`<key>.json` (URL, final URL, status, content type, time, SHA-256 of the body). A later run reads
the stored copy, so a report can be re-checked against exactly what was seen the first time.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import httpx

DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


@dataclass
class Fetched:
    url: str
    final_url: str
    status: int            # 0 when no HTTP response came back
    content_type: str
    fetched_at: str        # UTC, ISO 8601
    sha256: str            # of the body; empty when there is none
    body_path: str         # empty when there is no body
    error: str = ""
    from_cache: bool = False

    def body(self) -> bytes:
        if not self.body_path or not os.path.exists(self.body_path):
            return b""
        with open(self.body_path, "rb") as fh:
            return fh.read()


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def request_url(url: str) -> str:
    """Google's help pages answer in the visitor's language; ask for English so quotes match."""
    if "support.google.com" in url and "hl=" not in url:
        base = url.split("#")[0]
        return base + ("&" if "?" in base else "?") + "hl=en"
    return url


class Fetcher:
    def __init__(self, cache_dir: str, *, offline: bool = False, timeout: float = 40.0,
                 user_agent: str = DEFAULT_UA, client: httpx.Client | None = None):
        self.cache_dir = cache_dir
        self.offline = offline
        os.makedirs(cache_dir, exist_ok=True)
        self.client = client or httpx.Client(
            follow_redirects=True, timeout=timeout,
            headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"})

    def _paths(self, url: str) -> tuple[str, str]:
        key = cache_key(url)
        return os.path.join(self.cache_dir, key + ".body"), os.path.join(self.cache_dir, key + ".json")

    def cached(self, url: str) -> Fetched | None:
        _, meta = self._paths(url)
        if not os.path.exists(meta):
            return None
        with open(meta, encoding="utf-8") as fh:
            f = Fetched(**json.load(fh))
        f.from_cache = True
        if f.body_path:                     # the cache directory may have moved since
            f.body_path = self._paths(url)[0]
        return f

    def get(self, url: str) -> Fetched:
        hit = self.cached(url)
        if hit is not None:
            return hit
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.offline:
            return Fetched(url, url, 0, "", now, "", "", error="offline: not in cache")
        body_path, meta_path = self._paths(url)
        try:
            r = self.client.get(request_url(url))
            data = r.content
            with open(body_path, "wb") as fh:
                fh.write(data)
            f = Fetched(url, str(r.url), r.status_code, r.headers.get("content-type", ""), now,
                        hashlib.sha256(data).hexdigest(), body_path)
        except httpx.HTTPError as e:
            f = Fetched(url, url, 0, "", now, "", "", error=f"{type(e).__name__}: {e}"[:300])
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(f) | {"from_cache": False}, fh, indent=1)
        return f
