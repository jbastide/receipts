"""A fake web for the tests: no request leaves the machine."""
import json

import httpx
import pytest

from receipts.fetch import Fetcher

ARTICLE = """
<html><head><title>Call recording</title><script>var x = "not page text";</script></head>
<body><h1>Call recording for Local Services</h1>
<p>All phone calls, emails, and text-based communications delivered through Google may be monitored
or recorded to detect spam or unauthorized behavior and to improve quality.</p>
<p>Recordings are kept for 30 days and can be played back by the account owner. In legal categories
the retention period is 60 days, after which recordings are deleted automatically.</p>
<p>Businesses can turn off call recording in the settings page of their account at any time, and the
change applies to new calls only. Callers hear a short message before the call is connected.</p>
</body></html>
"""

OTHER = "<html><body><p>" + ("An unrelated page about gardening and soil, tomatoes and compost. " * 12) + "</p></body></html>"


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF with one line of text, written by hand."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


PDF_LINE = ("The committee concluded that recorded calls from advertising platforms raise questions "
            "of confidentiality for prospective clients. ") * 4


class FakeWeb:
    def __init__(self):
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        html = {"content-type": "text/html; charset=utf-8"}
        if url.startswith("https://example.org/article"):
            return httpx.Response(200, headers=html, text=ARTICLE)
        if url == "https://example.org/other":
            return httpx.Response(200, headers=html, text=OTHER)
        if url == "https://example.org/script-app":
            return httpx.Response(200, headers=html, text="<html><body><div id=app></div></body></html>")
        if url == "https://example.org/forbidden":
            return httpx.Response(403, headers=html, text="<html>Access denied</html>")
        if url in ("https://example.org/moved", "https://example.org/never"):
            return httpx.Response(404, headers=html, text="<html>Not found</html>")
        if url == "https://example.org/opinion.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=minimal_pdf(PDF_LINE))
        if url.startswith("https://web.archive.org/cdx/"):
            if "moved" in url:
                return httpx.Response(200, json=[["timestamp", "original"], ["20210304050607", "https://example.org/moved"]])
            return httpx.Response(200, json=[])
        if url.startswith("https://arctic-shift.photon-reddit.com/api/posts/ids"):
            return httpx.Response(200, json={"data": [{"title": "Does Google record calls to law firms?",
                                                       "selftext": "We noticed a recording notice on calls from our ads last month."}]})
        if url.startswith("https://arctic-shift.photon-reddit.com/api/comments/search"):
            return httpx.Response(200, json={"data": [{"body": "Yes, you can switch it off in the call settings, it took me five minutes to find. " * 6}]})
        if "support.google.com" in url:
            return httpx.Response(200, headers=html, text=ARTICLE if "hl=en" in url else OTHER)
        raise httpx.ConnectError("no such host", request=request)


@pytest.fixture
def web():
    return FakeWeb()


@pytest.fixture
def fetcher(tmp_path, web):
    client = httpx.Client(transport=httpx.MockTransport(web), follow_redirects=True)
    return Fetcher(str(tmp_path / "cache"), client=client)


@pytest.fixture
def make_fetcher(tmp_path, web):
    def make(offline=False):
        client = httpx.Client(transport=httpx.MockTransport(web), follow_redirects=True)
        return Fetcher(str(tmp_path / "cache"), client=client, offline=offline)
    return make


def dump(obj) -> str:
    return json.dumps(obj)
