"""docsite — the project's docs, rendered: served live at /docs and exportable.

Two consumers, one renderer:
  - the supervisor serves `GET /docs` (index) and `GET /docs/<slug>` so a browser on
    the LAN can read the docs next to the playground, no checkout needed;
  - `python -m umweltd.docsite --export <dir>` writes the same pages as a
    self-contained static site (inline CSS, no external assets) you can zip and send.

The renderer is a deliberately small markdown subset (headers, emphasis, code,
fenced blocks, lists, tables, blockquotes, links, rules) — enough for this repo's
docs, HTML-escaped first so a doc can never inject markup. Docs are read from the
repo checkout (found by walking up from this package, or UMWELT_DOCS_DIR); an
installed-package deployment without a checkout serves a clear "docs not bundled"
page instead of failing.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

# slug -> (title, repo-relative path). Curated, ordered — the index follows this.
DOC_REGISTRY: "tuple[tuple[str, str, str], ...]" = (
    ("overview", "umwelt, in plain terms", "docs/overview/README.md"),
    ("what-it-can-do", "What it can actually do", "docs/overview/what-it-can-do.md"),
    ("working-with-it", "How you'd work with it", "docs/overview/working-with-it.md"),
    ("readme", "README (technical)", "README.md"),
    ("claims", "CLAIMS — the evidence ledger", "CLAIMS.md"),
    ("forge", "The forge — rant to running world", "docs/FORGE.md"),
    ("service", "umweltd — the engine as a service", "docs/SERVICE.md"),
    ("new-domain", "Starting a new domain", "docs/NEW_DOMAIN.md"),
    ("spec", "The DomainSpec reference", "docs/SPEC.md"),
    ("changelog", "Changelog", "CHANGELOG.md"),
)

CSS = """
:root { color-scheme: dark; }
body { background:#101418; color:#d7dde3; font:16px/1.6 system-ui,sans-serif;
       max-width:56rem; margin:0 auto; padding:2rem 1.2rem 4rem; }
a { color:#6fc3ff; text-decoration:none; } a:hover { text-decoration:underline; }
h1,h2,h3 { line-height:1.25; color:#f0f4f8; margin:1.6em 0 .5em; }
h1 { font-size:1.7rem; border-bottom:1px solid #2a3540; padding-bottom:.4rem; }
h2 { font-size:1.3rem; } h3 { font-size:1.1rem; }
code { background:#1b232c; border-radius:4px; padding:.1em .35em;
       font:.9em/1.4 ui-monospace,monospace; color:#a8d8a8; }
pre { background:#151b22; border:1px solid #26313c; border-radius:8px;
      padding:.9rem 1rem; overflow-x:auto; }
pre code { background:none; padding:0; }
blockquote { border-left:3px solid #3b82c4; margin:.8em 0; padding:.1em 1em;
             color:#a9b4bf; }
table { border-collapse:collapse; margin:1em 0; display:block; overflow-x:auto; }
th,td { border:1px solid #2a3540; padding:.4em .7em; text-align:left;
        vertical-align:top; }
th { background:#1b232c; }
hr { border:0; border-top:1px solid #2a3540; margin:2em 0; }
.crumb { color:#8b98a5; font-size:.9rem; margin-bottom:1.5rem; }
.crumb a { color:#8b98a5; }
ul.doc-index { list-style:none; padding:0; }
ul.doc-index li { margin:.55em 0; }
ul.doc-index .path { color:#5c6a77; font-size:.85em; margin-left:.6em; }
"""

_INLINE_PATTERNS = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*]+)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"), r'<a href="\2">\1</a>'),
)


def _inline(text: str) -> str:
    for pat, rep in _INLINE_PATTERNS:
        text = pat.sub(rep, text)
    return text


def _rewrite_link(href: str) -> str:
    """Point intra-repo markdown links at their /docs slug when we serve that doc;
    external links pass through; everything else degrades to plain text-ish '#'."""
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return href
    clean = re.sub(r"^(\.\./)+|^\./", "", href.split("#")[0])
    for slug, _title, rel in DOC_REGISTRY:
        if rel.endswith(clean) and clean:
            return slug
    return "#"


def render_markdown(md: str) -> str:
    """The subset renderer. Escape first; markup is reintroduced deliberately."""
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    in_list: "str | None" = None            # "ul" | "ol"

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    while i < len(lines):
        raw = lines[i]
        line = html.escape(raw, quote=False)

        if raw.startswith("```"):
            close_list()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(html.escape(lines[i], quote=False))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            close_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if re.match(r"^(\s*[-*_]){3,}\s*$", raw):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        if raw.startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            close_list()
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<table><tr>" +
                       "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in
                         html.escape(lines[i], quote=False).strip().strip("|").split("|")]
                out.append("<tr>" +
                           "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
            continue

        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            if in_list != "ul":
                close_list()
                out.append("<ul>")
                in_list = "ul"
            item = [m.group(1)]
            i += 1
            while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                    and not re.match(r"^\s*[-*]\s+", lines[i]):
                item.append(html.escape(lines[i].strip(), quote=False))
                i += 1
            out.append(f"<li>{_inline(' '.join(item))}</li>")
            continue

        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            if in_list != "ol":
                close_list()
                out.append("<ol>")
                in_list = "ol"
            item = [m.group(1)]
            i += 1
            while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                    and not re.match(r"^\s*(\d+\.|[-*])\s+", lines[i]):
                item.append(html.escape(lines[i].strip(), quote=False))
                i += 1
            out.append(f"<li>{_inline(' '.join(item))}</li>")
            continue

        if raw.startswith(">"):
            close_list()
            quote: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(html.escape(lines[i].lstrip("> "), quote=False))
                i += 1
            out.append("<blockquote><p>" + _inline(" ".join(quote)) +
                       "</p></blockquote>")
            continue

        if not raw.strip():
            close_list()
            i += 1
            continue

        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() \
                and not re.match(r"^(#{1,4}\s|```|\||\s*[-*]\s|\s*\d+\.\s|>)", lines[i]):
            para.append(html.escape(lines[i], quote=False))
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")

    close_list()
    page = "\n".join(out)
    # Rewrite intra-repo links to served slugs (after inline pass built the <a>s).
    return re.sub(r'href="([^"]+)"', lambda m: f'href="{_rewrite_link(m.group(1))}"',
                  page)


def find_docs_root() -> "Path | None":
    """The repo checkout holding docs/ + README.md: UMWELT_DOCS_DIR wins, else walk
    up from this package (editable installs land in <repo>/src/umweltd)."""
    env = os.environ.get("UMWELT_DOCS_DIR")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs").is_dir() and (parent / "README.md").is_file():
            return parent
    return None


def _page(title: str, body: str, *, crumb: bool = True) -> str:
    nav = ('<div class="crumb"><a href="/ui">playground</a> · '
           '<a href="/docs">docs index</a></div>' if crumb else "")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{nav}{body}</body></html>")


def render_index() -> str:
    root = find_docs_root()
    items = []
    for slug, title, rel in DOC_REGISTRY:
        exists = root is not None and (root / rel).is_file()
        if exists:
            items.append(f'<li><a href="{slug}">{html.escape(title)}</a>'
                         f'<span class="path">{rel}</span></li>')
    if not items:
        body = ("<h1>umwelt docs</h1><p>No docs found — this deployment runs from "
                "the installed package without a repo checkout. Set "
                "<code>UMWELT_DOCS_DIR</code> to a checkout's root, or read the "
                "docs in the repository.</p>")
        return _page("umwelt docs", body)
    return _page("umwelt docs",
                 "<h1>umwelt docs</h1><ul class='doc-index'>" + "".join(items) +
                 "</ul>")


def render_doc(slug: str) -> "str | None":
    root = find_docs_root()
    if root is None:
        return None
    for s, title, rel in DOC_REGISTRY:
        if s == slug:
            path = root / rel
            if not path.is_file():
                return None
            return _page(title, render_markdown(path.read_text()))
    return None


def export_site(dest: "Path | str") -> "list[str]":
    """Write the whole doc set as a static site (index.html + <slug>.html, links
    rewritten to .html). Returns the written filenames."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    index = render_index().replace('href="/ui">playground</a> · ', "")
    for slug, _t, _r in DOC_REGISTRY:
        index = index.replace(f'href="{slug}"', f'href="{slug}.html"')
    (dest / "index.html").write_text(index.replace('href="/docs"', 'href="index.html"'))
    written.append("index.html")
    for slug, _title, _rel in DOC_REGISTRY:
        page = render_doc(slug)
        if page is None:
            continue
        for s2, _t2, _r2 in DOC_REGISTRY:
            page = page.replace(f'href="{s2}"', f'href="{s2}.html"')
        page = page.replace('href="/docs"', 'href="index.html"') \
                   .replace('<a href="/ui">playground</a> · ', "")
        (dest / f"{slug}.html").write_text(page)
        written.append(f"{slug}.html")
    return written


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m umweltd.docsite",
                                 description="export the umwelt docs as a static "
                                             "HTML site")
    ap.add_argument("--export", required=True, metavar="DIR",
                    help="destination directory")
    args = ap.parse_args(argv)
    written = export_site(args.export)
    print(f"wrote {len(written)} pages to {args.export}")
    return 0 if len(written) > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
