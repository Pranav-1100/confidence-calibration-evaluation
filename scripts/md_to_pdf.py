#!/usr/bin/env python3
"""
Render writeups/PAPER_DRAFT.md -> paper-styled HTML -> PDF via headless Chrome.
Usage: .venv-figs/bin/python scripts/md_to_pdf.py [input.md] [output.pdf]
"""
import os, subprocess, sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "writeups", "PAPER_DRAFT.md")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "writeups", "PAPER_DRAFT.pdf")
HTML = os.path.splitext(OUT)[0] + ".render.html"

body = markdown.markdown(
    open(IN, encoding="utf-8").read(),
    extensions=["tables", "fenced_code", "sane_lists", "smarty"],
)

css = """
@page { size: A4; margin: 22mm 20mm; }
* { -webkit-print-color-adjust: exact; }
body { font-family: Charter, Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; max-width: 100%; margin: 0; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 10px; }
h2 { font-size: 13pt; margin: 22px 0 8px; border-bottom: 1px solid #999; padding-bottom: 3px; }
h3 { font-size: 11.5pt; margin: 16px 0 6px; }
p, li { text-align: justify; }
strong { color: #000; }
blockquote { border-left: 3px solid #bbb; margin: 10px 0; padding: 2px 14px;
             background: #f7f7f5; font-size: 9.5pt; }
code { font-family: 'SF Mono', Menlo, monospace; font-size: 8.6pt; background: #f2f2ef;
       padding: 1px 3px; border-radius: 2px; }
pre { background: #f2f2ef; border: 1px solid #ddd; border-radius: 4px; padding: 9px 12px;
      overflow-x: hidden; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 12px auto; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 10px; text-align: center; }
th { background: #efefec; }
img { max-width: 88%; display: block; margin: 14px auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
h2, h3 { page-break-after: avoid; }
table, blockquote, pre { page-break-inside: avoid; }
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
