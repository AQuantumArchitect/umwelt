"""The playground + docsite surface: static pages load WITHOUT auth (they hold no
world data), every JSON route stays behind auth, UMWELTD_UI=off kills the surface,
and the docs render/export sanely."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from umweltd.docsite import (DOC_REGISTRY, export_site, find_docs_root,
                             render_doc, render_index, render_markdown)
from umweltd.supervisor import Supervisor, _Handler


# ── the markdown renderer ─────────────────────────────────────────────────────────

def test_render_markdown_basics():
    h = render_markdown("# Title\n\nsome **bold** and `code` and *stress*.\n")
    assert "<h1>Title</h1>" in h
    assert "<strong>bold</strong>" in h and "<code>code</code>" in h
    assert "<em>stress</em>" in h


def test_render_markdown_escapes_html():
    h = render_markdown("hello <script>alert(1)</script>\n")
    assert "<script>" not in h and "&lt;script&gt;" in h


def test_render_markdown_fence_list_table_quote():
    md = ("```python\nx = 1 < 2\n```\n\n- one\n- two\n\n1. first\n2. second\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n> quoted\n")
    h = render_markdown(md)
    assert "<pre><code>x = 1 &lt; 2</code></pre>" in h
    assert h.count("<li>") == 4 and "<ul>" in h and "<ol>" in h
    assert "<table>" in h and "<th>a</th>" in h and "<td>2</td>" in h
    assert "<blockquote>" in h


def test_intra_repo_links_rewrite_to_slugs():
    h = render_markdown("see [the ledger](../../CLAIMS.md) and "
                        "[the web](https://example.com) and "
                        "[unserved](src/umwelt/boot.py)\n")
    assert 'href="claims"' in h
    assert 'href="https://example.com"' in h
    assert 'href="#"' in h                   # unserved repo file degrades


# ── the doc registry against THIS repo ────────────────────────────────────────────

def test_registry_docs_exist_and_render():
    root = find_docs_root()
    assert root is not None, "docs root not found from the repo checkout"
    for slug, _title, rel in DOC_REGISTRY:
        assert (root / rel).is_file(), f"registry points at missing {rel}"
        page = render_doc(slug)
        assert page and "<h1" in page, f"{slug} rendered empty"
    assert "doc-index" in render_index()


def test_export_site_writes_standalone_pages(tmp_path):
    written = export_site(tmp_path)
    assert "index.html" in written and len(written) == len(DOC_REGISTRY) + 1
    index = (tmp_path / "index.html").read_text()
    assert 'href="overview.html"' in index
    assert "/docs" not in index              # fully static, no server routes
    forge = (tmp_path / "forge.html").read_text()
    assert "<style>" in forge                # self-contained styling


# ── the served surface ────────────────────────────────────────────────────────────

@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("UMWELTD_HOME", str(tmp_path))
    monkeypatch.delenv("UMWELTD_UI", raising=False)
    _Handler.sup = Supervisor()
    _Handler.api_key = "secret"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    _Handler.api_key = None


def _get(url, key=None):
    req = urllib.request.Request(url,
                                 headers={"X-API-Key": key} if key else {})
    return urllib.request.urlopen(req, timeout=10)


def test_static_surface_loads_without_key_but_api_stays_locked(server):
    with _get(server + "/ui") as r:
        assert r.status == 200 and "text/html" in r.headers["Content-Type"]
        assert b"umwelt playground" in r.read()
    with _get(server + "/docs") as r:
        assert r.status == 200 and b"umwelt docs" in r.read()
    with _get(server + "/docs/overview") as r:
        assert r.status == 200 and b"plain terms" in r.read()
    # the JSON API is still 401 without the key…
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + "/worlds")
    assert e.value.code == 401
    # …and works with it.
    with _get(server + "/worlds", key="secret") as r:
        assert json.loads(r.read()) == []


def test_root_redirects_to_ui(server):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as e:
        opener.open(server + "/", timeout=10)
    assert e.value.code == 302 and e.value.headers["Location"] == "/ui"


def test_unknown_doc_404s(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + "/docs/no-such-doc")
    assert e.value.code == 404


def test_ui_off_kills_the_static_surface(server, monkeypatch):
    monkeypatch.setenv("UMWELTD_UI", "off")
    for path in ("/ui", "/docs"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(server + path)
        assert e.value.code == 401           # falls through to the authed API: no route without key
