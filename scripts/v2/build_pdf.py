#!/usr/bin/env python3
"""
Render paper.md -> an arXiv-style preprint PDF via headless Chrome.

Why this exists rather than reusing scripts/md_to_pdf.py (which built v1):
  * That script styles only the FIRST paragraph after the Abstract heading
    (`.front h2 + p`). v2's abstract is three paragraphs, so paragraphs two and three
    fell out of the abstract block and rendered centered and ragged.
  * Its `.twocol` class never sets `column-count`, so the "two-column" body was
    always one column anyway. v2 carries 11 tables, one of them seven columns wide,
    plus six figures. Two columns would break all of them, so this renders a single
    column deliberately and sizes the measure for readability instead.

Usage:
    <venv>/python scripts/v2/build_pdf.py [in.md] [out.pdf]

Requires: `markdown` (pip) and Google Chrome at the standard macOS path.
"""
import os, re, subprocess, sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IN   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper.md")
OUT  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "paper.pdf")
HTML = os.path.splitext(OUT)[0] + ".render.html"

src = open(IN, encoding="utf-8").read()

# Figure images are referenced relative to the markdown file; make them absolute so
# headless Chrome resolves them from the temp HTML location.
src = src.replace("](figures/", f"]({os.path.join(ROOT, 'figures')}/")

html = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists", "smarty"])

# Split front matter (title/byline/abstract/Figure 1) from the numbered body.
MARK = "<h2>1. Introduction</h2>"
if MARK in html:
    front, rest = html.split(MARK, 1)
    html = f"<div class='front'>{front}</div><div class='body'>{MARK}{rest}</div>"
else:
    print("WARNING: '1. Introduction' heading not found; rendering as one block")
    html = f"<div class='body'>{html}</div>"

# Figure captions are bold-led paragraphs starting "Figure N:" - tag them so they can
# be styled small and kept with their image.
html = re.sub(r"<p><strong>(Figure \d+:.*?)</strong>", r"<p class='caption'><strong>\1</strong>", html)

CSS = """
@page { size: A4; margin: 20mm 20mm 18mm; @bottom-center { content: counter(page); } }
* { -webkit-print-color-adjust: exact; box-sizing: border-box; }
html { -webkit-font-smoothing: antialiased; }
body { font-family: Charter, Georgia, 'Times New Roman', serif;
       font-size: 10.2pt; line-height: 1.5; color: #111; margin: 0; }

/* ---------------- front matter ---------------- */
.front { margin-bottom: 6mm; }
.front h1 { font-size: 17pt; line-height: 1.28; text-align: center;
            margin: 2mm 6mm 5mm; font-weight: 700; letter-spacing: -0.15px; }
.front > p { text-align: center; margin: 2px 0; font-size: 10pt; }
.front > p em { color: #444; font-size: 9.2pt; }
.front h2 { text-align: center; font-size: 11pt; font-variant: small-caps;
            letter-spacing: 0.6px; border: none; margin: 7mm 0 3mm; }
/* every paragraph of the abstract, not just the first */
.front h2 ~ p { text-align: justify; margin: 0 14mm 2.6mm; font-size: 9.7pt;
                line-height: 1.5; hyphens: auto; }
.front hr { display: none; }

/* ---------------- body ---------------- */
.body { text-align: justify; hyphens: auto; }
.body h2 { font-size: 12.6pt; margin: 7mm 0 2.5mm; padding-bottom: 1.4mm;
           border-bottom: 0.7px solid #999; break-after: avoid; font-weight: 700; }
.body h3 { font-size: 10.9pt; margin: 4.5mm 0 1.8mm; break-after: avoid; font-weight: 700; }
.body p { margin: 0 0 2.6mm; orphans: 3; widows: 3; }
.body ul, .body ol { margin: 0 0 3mm 6mm; padding-left: 4mm; }
.body li { margin-bottom: 1.2mm; }
strong { color: #000; font-weight: 700; }

/* ---------------- figures and captions ---------------- */
img { max-width: 100%; display: block; margin: 4mm auto 1.5mm; break-inside: avoid; }
p.caption { font-size: 8.9pt; line-height: 1.42; color: #333; margin: 0 6mm 5mm;
            text-align: justify; break-before: avoid; }

/* ---------------- tables ---------------- */
table { border-collapse: collapse; margin: 3.5mm auto 4.5mm; font-size: 8.7pt;
        max-width: 100%; break-inside: avoid; }
th, td { border: 0.5px solid #aaa; padding: 1.5mm 2.4mm; text-align: center;
         line-height: 1.32; }
th { background: #eeeeea; font-weight: 700; }
td:first-child, th:first-child { text-align: left; }

/* ---------------- code, quotes, rules ---------------- */
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 8.3pt;
       background: #f2f2ef; padding: 0 1.5px; border-radius: 2px; }
pre { background: #f6f6f3; border: 0.6px solid #ddd; border-radius: 3px;
      padding: 2.5mm 3mm; white-space: pre-wrap; word-wrap: break-word;
      font-size: 8.2pt; line-height: 1.4; break-inside: avoid; margin: 3mm 0; }
pre code { background: none; padding: 0; }
blockquote { border-left: 2.5px solid #bbb; margin: 3mm 0; padding: 1.5mm 4mm;
             background: #f8f8f6; font-size: 9.2pt; break-inside: avoid; }
hr { border: none; border-top: 0.6px solid #ccc; margin: 5mm 0; }
a { color: #14396b; text-decoration: none; }
"""

open(HTML, "w", encoding="utf-8").write(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Calibrated Enough to Know, Not Calibrated to Act</title>"
    f"<style>{CSS}</style></head><body>{html}</body></html>"
)

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if not os.path.exists(chrome):
    sys.exit(f"Chrome not found at {chrome}")
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={os.path.abspath(OUT)}", "--print-to-pdf-no-header",
                f"file://{os.path.abspath(HTML)}"],
               check=True, capture_output=True)
os.remove(HTML)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
