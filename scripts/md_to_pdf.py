#!/usr/bin/env python3
"""
Render writeups/PAPER_DRAFT.md -> arXiv-preprint-styled PDF via headless Chrome.
Layout: centered title/author block + full-width abstract, then TWO-COLUMN body
(from '1. Introduction' onward), A4.

Usage: .venv-figs/bin/python scripts/md_to_pdf.py [input.md] [output.pdf]
"""
import os, subprocess, sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "paper.md")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper.pdf")
HTML = os.path.splitext(OUT)[0] + ".render.html"

body = markdown.markdown(
    open(IN, encoding="utf-8").read(),
    extensions=["tables", "fenced_code", "sane_lists", "smarty"],
)

# split: front matter (title/author/abstract) full-width, rest two-column
MARK = None
for probe in ("<h2>1. Introduction</h2>", "<h2>1. Introduction</h2>"):
    if probe in body: MARK = probe; break
if MARK:
    front, rest = body.split(MARK, 1)
    body = f"<div class='front'>{front}</div><div class='twocol'>{MARK}{rest}</div>"
else:
    body = f"<div class='twocol'>{body}</div>"

css = """
@page { size: A4; margin: 21mm 19mm; }
* { -webkit-print-color-adjust: exact; box-sizing: border-box; }
body { font-family: Charter, Georgia, 'Times New Roman', serif; font-size: 10.3pt;
       line-height: 1.45; color: #111; margin: 0; }
/* ---------- front matter (centered, full width) ---------- */
.front h1 { font-size: 16.5pt; line-height: 1.3; text-align: center; margin: 4px 8mm 14px; }
.front p { text-align: center; margin: 3px 0; font-size: 10pt; }
.front h2 { text-align: center; font-size: 11pt; border: none; margin: 16px 0 6px;
            text-transform: none; }
.front h2 + p { text-align: justify; margin: 0 12mm; font-size: 9.8pt; line-height: 1.45; }
.front hr { border: none; margin: 6px 0; }
/* ---------- two-column body ---------- */
.twocol { text-align: justify; }
.twocol h2 { font-size: 12.5pt; margin: 20px 0 8px; border-bottom: 0.6px solid #888;
             padding-bottom: 3px; break-after: avoid; }
.twocol h3 { font-size: 11pt; margin: 14px 0 5px; break-after: avoid; }
.twocol p, .twocol li { text-align: justify; }
strong { color: #000; }
blockquote { border-left: 2.5px solid #bbb; margin: 10px 0; padding: 3px 10px;
             background: #f7f7f5; font-size: 9pt; break-inside: avoid; }
code { font-family: 'SF Mono', Menlo, monospace; font-size: 8.4pt; background: #f2f2ef;
       padding: 0 2px; border-radius: 2px; }
pre { background: #f2f2ef; border: 1px solid #ddd; border-radius: 3px; padding: 6px 8px;
      white-space: pre-wrap; word-wrap: break-word; font-size: 8.2pt; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 12px auto; font-size: 9.3pt; max-width: 100%;
        break-inside: avoid; }
th, td { border: 0.6px solid #999; padding: 4px 9px; text-align: center; }
th { background: #efefec; }
img { max-width: 86%; display: block; margin: 14px auto; break-inside: avoid; }
hr { border: none; border-top: 0.6px solid #ccc; margin: 12px 0; }
a { color: #1a3c6e; text-decoration: none; }
"""

open(HTML, "w", encoding="utf-8").write(
    f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
)

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={os.path.abspath(OUT)}", f"file://{os.path.abspath(HTML)}"],
               check=True, capture_output=True, timeout=120)
os.remove(HTML)
print("wrote", os.path.abspath(OUT))
