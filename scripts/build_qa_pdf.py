#!/usr/bin/env python3
"""Build the judge-facing model/system PDF from docs/MODEL_AND_SYSTEM_QA.md.

Run from the repo root (dev-only dependencies; they are not app runtime deps):

    uv run --no-project --with markdown --with weasyprint \
        python scripts/build_qa_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "MODEL_AND_SYSTEM_QA.md"
OUTPUT = ROOT / "docs" / "Aedes_Sentinel_Model_System_Design.pdf"

CSS = r"""
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @top-left {
    content: "Aedes Sentinel  ·  Team 3AM";
    color: #6B7A70;
    font-size: 7.4pt;
    font-family: "DejaVu Sans", sans-serif;
    letter-spacing: .02em;
  }
  @top-right {
    content: "Model & System Design  ·  Phase 2";
    color: #6B7A70;
    font-size: 7.4pt;
    font-family: "DejaVu Sans", sans-serif;
  }
  @bottom-left {
    content: "version-onsite-final  ·  local-first  ·  0 runtime APIs";
    color: #8A968E;
    font-size: 7pt;
    font-family: "DejaVu Sans", sans-serif;
  }
  @bottom-right {
    content: "Page " counter(page) " of " counter(pages);
    color: #8A968E;
    font-size: 7pt;
    font-family: "DejaVu Sans", sans-serif;
  }
}

@page cover {
  margin: 0;
  background: #1A1A1A;
  @top-left { content: ""; }
  @top-right { content: ""; }
  @bottom-left { content: ""; }
  @bottom-right { content: ""; }
}

html { font-family: "DejaVu Sans", sans-serif; color: #2B2B2B; }
body { margin: 0; }

.cover {
  page: cover;
  box-sizing: border-box;
  height: 297mm;
  padding: 28mm 22mm 20mm;
  color: #FFFFFF;
  background: #1A1A1A;
}

.cover h1 { page-break-before: avoid; }

.cover-top { margin-bottom: 28mm; }
.cover-mark {
  display: inline-block;
  background: #FFD84D;
  color: #1A1A1A;
  font-weight: 800;
  font-size: 18pt;
  letter-spacing: -.04em;
  padding: 4px 9px;
  border-radius: 5px;
  margin-right: 9px;
  vertical-align: middle;
}
.cover-kicker {
  display: inline-block;
  color: #FFD84D;
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
  vertical-align: middle;
}
.cover-title {
  color: #FFFFFF;
  font-size: 40pt;
  line-height: .98;
  letter-spacing: -.045em;
  margin: 0 0 6px;
  font-weight: 800;
}
.cover-sub {
  color: #FFD84D;
  font-size: 17pt;
  line-height: 1.2;
  margin: 0 0 8mm;
  font-weight: 600;
}
.cover-rule {
  width: 34mm;
  height: 3px;
  background: #FFD84D;
  margin: 7mm 0;
}
.cover-lede {
  color: #D7DDD8;
  font-size: 10.5pt;
  line-height: 1.55;
  max-width: 132mm;
  margin: 0 0 16mm;
}
.cover-grid {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-top: 1px solid #3B443D;
  border-bottom: 1px solid #3B443D;
  padding: 6mm 0;
  margin-bottom: 10mm;
}
.cover-grid > div {
  display: table-cell;
  width: 25%;
  vertical-align: top;
  padding-right: 3mm;
}
.cover-grid span {
  display: block;
  color: #8F9B92;
  font-size: 7pt;
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: 3px;
}
.cover-grid strong {
  color: #FFFFFF;
  font-size: 10pt;
  font-weight: 700;
}
.cover-foot {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-top: 1px solid #3B443D;
  padding-top: 5mm;
}
.cover-foot > div {
  display: table-cell;
  width: 33.33%;
  vertical-align: top;
  padding-right: 4mm;
}
.cover-label {
  display: block;
  color: #8F9B92;
  font-size: 6.8pt;
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: 2px;
}
.cover-value {
  color: #FFFFFF;
  font-size: 8pt;
  font-weight: 600;
}
.cover-subvalue {
  color: #9EAAA2;
  font-size: 7.2pt;
  font-weight: 500;
}

.pagebreak { page-break-after: always; }

.toc-title {
  color: #1F5A34;
  font-size: 24pt;
  font-weight: 800;
  letter-spacing: -.03em;
  margin: 0 0 8mm;
  padding-bottom: 3mm;
  border-bottom: 3px solid #FFD84D;
}
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc li {
  margin: 0 0 4.5mm;
  padding-bottom: 2.5mm;
  border-bottom: 1px solid #E2E8E3;
  font-size: 10.5pt;
}
.toc a { color: #1A1A1A; text-decoration: none; font-weight: 600; }
.toc a::after {
  content: "   ·   p. " target-counter(attr(href), page);
  color: #7A877E;
  font-weight: 400;
  font-size: 9pt;
}

h1 {
  page-break-before: always;
  color: #1F5A34;
  font-size: 21pt;
  line-height: 1.08;
  letter-spacing: -.035em;
  margin: 0 0 5mm;
  padding-bottom: 2.5mm;
  border-bottom: 3px solid #FFD84D;
}
h1.keep-with-previous { page-break-before: avoid; }
h2 {
  color: #1A1A1A;
  font-size: 13.5pt;
  line-height: 1.25;
  letter-spacing: -.02em;
  margin: 7mm 0 2.5mm;
  padding-left: 3mm;
  border-left: 4px solid #1F5A34;
}
h3 {
  color: #1F5A34;
  font-size: 10.5pt;
  margin: 5mm 0 1.5mm;
}
p, li { font-size: 9.4pt; line-height: 1.5; }
p { margin: 0 0 2.8mm; }
ul, ol { margin: 0 0 3mm 5mm; padding: 0; }
li { margin-bottom: 1.2mm; }

blockquote {
  background: #FFF8D6;
  border-left: 4px solid #FFD84D;
  margin: 4mm 0;
  padding: 3.5mm 4mm;
  color: #3A3A2E;
  page-break-inside: avoid;
}
blockquote p { margin: 0; font-size: 9.4pt; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 5mm;
  font-size: 8.1pt;
  line-height: 1.35;
}
th {
  background: #1F5A34;
  color: #FFFFFF;
  text-align: left;
  font-size: 7.5pt;
  letter-spacing: .02em;
  padding: 2.2mm 2mm;
  vertical-align: bottom;
}
td {
  border-bottom: 1px solid #DDE5DF;
  padding: 1.9mm 2mm;
  vertical-align: top;
}
tr:nth-child(even) td { background: #F6F8F5; }
tr { page-break-inside: avoid; }

pre {
  background: #F3F5F3;
  border-left: 3px solid #1F5A34;
  margin: 3mm 0 5mm;
  padding: 3mm 3.5mm;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 7.1pt;
  line-height: 1.3;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 7.8pt;
  background: #EEF2EE;
  padding: .5mm 1mm;
  border-radius: 1mm;
}
pre code { background: transparent; padding: 0; font-size: 7.1pt; }

.closing {
  margin-top: 8mm;
  background: #1A1A1A;
  color: #FFFFFF;
  border-radius: 3mm;
  padding: 5mm;
  page-break-inside: avoid;
}
.closing-mark {
  display: inline-block;
  background: #FFD84D;
  color: #1A1A1A;
  font-weight: 800;
  font-size: 11pt;
  padding: 2mm 2.5mm;
  border-radius: 2mm;
  margin-right: 3mm;
  vertical-align: middle;
}
.closing div { display: inline-block; vertical-align: middle; }
.closing strong { color: #FFD84D; }
"""


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"missing source: {SOURCE}")
    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"toc_depth": "1"}},
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aedes Sentinel — Model & System Design</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(OUTPUT)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1_000_000:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
