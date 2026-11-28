"""Convert report/report.md (a restricted Markdown subset) into a styled DOCX.

Supported: #/##/### headings, paragraphs, bullet lists, pipe tables, images with
captions, blockquotes, fenced code, **bold**, `code`, and <!-- pagebreak -->.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "SynthAudit_报告.docx"

BODY_FONT_CN = "宋体"
HEAD_FONT_CN = "黑体"
LATIN_FONT = "Times New Roman"


def set_run_font(run, size=None, bold=None, color=None, cn=BODY_FONT_CN, latin=LATIN_FONT):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cn)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade(cell, hex_color: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`)")


def add_inline(paragraph, text: str, size=10.5, bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=size, bold=True)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            run.font.size = Pt(size - 1)
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size, bold=bold)


def new_document() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CN)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(4)

    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        st = doc.styles[name]
        st.font.name = HEAD_FONT_CN
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0x1F, 0x24, 0x2B)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT_CN)
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)
    for run in footer.runs:
        run.font.size = Pt(8)
    return doc


def add_title_page(doc: Document, title: str, subtitle: str):
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    set_run_font(run, size=19.5, bold=True, cn=HEAD_FONT_CN)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(subtitle)
    set_run_font(run, size=13, color=RGBColor(0x44, 0x4A, 0x52), cn=HEAD_FONT_CN)
    p.paragraph_format.space_after = Pt(26)

    for _ in range(4):
        doc.add_paragraph()
    for line in [
        "课程：机器学习 / 数据分析项目",
        "题目：生成式表格数据增强的标签泄漏审计",
        "代码：https://github.com/Croissant-cat/synthaudit-tabular",
        "日期：2026-11-22",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        set_run_font(run, size=11)
        p.paragraph_format.space_after = Pt(6)
    doc.add_page_break()


def add_paragraph(doc: Document, text: str, indent=True, size=10.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    add_inline(p, text, size=size)
    return p


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return
    n_cols = len(rows[0])
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    hdr = table.rows[0].cells
    for j, val in enumerate(rows[0]):
        hdr[j].text = ""
        p = hdr[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(val))
        set_run_font(run, size=8.5, bold=True)
        shade(hdr[j], "E8EDF3")
    for row in rows[1:]:
        cells = table.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            add_inline(p, str(val), size=8.3)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_image(doc: Document, alt: str, path: Path):
    if not path.exists():
        p = doc.add_paragraph()
        add_inline(p, f"[missing figure: {path}]", size=9)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(15.6))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_inline(cap, alt, size=8.8)
    for r in cap.runs:
        r.font.color.rgb = RGBColor(0x55, 0x5B, 0x63)
    cap.paragraph_format.space_after = Pt(8)


def parse_table(lines: list[str], start: int):
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip().strip("|")
        cells = [c.strip() for c in line.split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def main():
    text = REPORT_MD.read_text(encoding="utf-8")
    lines = text.splitlines()

    # first H1 = title, second non-empty line may be subtitle
    title = "生成式表格数据增强的效用审计"
    subtitle = "——数据层、方法层与分析层的端到端实验"
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            start = idx + 1
            break

    doc = new_document()
    add_title_page(doc, title, subtitle)

    i = start
    in_code = False
    code_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.6)
                run = p.add_run("\n".join(code_buf))
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                code_buf = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not s:
            i += 1
            continue
        if s == "<!-- pagebreak -->":
            doc.add_page_break()
            i += 1
            continue
        if s.startswith("#### "):
            p = doc.add_paragraph()
            run = p.add_run(s[5:])
            set_run_font(run, size=10.5, bold=True, cn=HEAD_FONT_CN)
            i += 1
            continue
        if s.startswith("### "):
            doc.add_heading(s[4:], level=3)
            i += 1
            continue
        if s.startswith("## "):
            doc.add_heading(s[3:], level=2)
            i += 1
            continue
        if s.startswith("# "):
            doc.add_heading(s[2:], level=1)
            i += 1
            continue
        if s.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.right_indent = Cm(0.8)
            add_inline(p, s[2:], size=9.5)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x44, 0x4A, 0x52)
            i += 1
            continue
        if s.startswith("!["):
            m = re.match(r"!\[(.+?)\]\((.+?)\)", s)
            if m:
                add_image(doc, m.group(1), ROOT / m.group(2))
            i += 1
            continue
        if s.startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(doc, rows)
            continue
        if s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, s[2:], size=10.5)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue
        if re.match(r"^\d+\.\s", s):
            p = doc.add_paragraph(style="List Number")
            add_inline(p, re.sub(r"^\d+\.\s", "", s), size=10.5)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        # regular paragraph (join consecutive lines)
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "- ", "![", "> ", "```"))
                    or re.match(r"^\d+\.\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        add_paragraph(doc, "".join(buf))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
