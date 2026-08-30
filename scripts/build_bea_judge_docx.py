from __future__ import annotations

import datetime as dt
import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "bea_judge_manuscript"
MANUSCRIPT = PAPER_DIR / "manuscript.md"
OUT_DOCX = PAPER_DIR / "bea_judge_manuscript.docx"


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def xesc(value: object) -> str:
    return html.escape(str(value), quote=True)


def para_props(style: str | None = None, align: str | None = None, first_line: int | None = None) -> str:
    parts: list[str] = []
    if style:
        parts.append(f'<w:pStyle w:val="{xesc(style)}"/>')
    if align:
        parts.append(f'<w:jc w:val="{xesc(align)}"/>')
    if first_line is not None:
        parts.append(f'<w:ind w:firstLine="{first_line}"/>')
    return f"<w:pPr>{''.join(parts)}</w:pPr>" if parts else ""


def run_text(text: str, bold: bool = False, italic: bool = False) -> str:
    rpr: list[str] = []
    if bold:
        rpr.append("<w:b/><w:bCs/>")
    if italic:
        rpr.append("<w:i/><w:iCs/>")
    props = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{xesc(text)}</w:t></w:r>'


def parse_inline_markdown(text: str) -> list[tuple[str, bool, bool]]:
    runs: list[tuple[str, bool, bool]] = []
    pos = 0
    for match in re.finditer(r"\*\*(.+?)\*\*", text):
        if match.start() > pos:
            runs.append((text[pos : match.start()], False, False))
        runs.append((match.group(1), True, False))
        pos = match.end()
    if pos < len(text):
        runs.append((text[pos:], False, False))
    return runs or [("", False, False)]


def paragraph(
    text: str = "",
    style: str | None = None,
    align: str | None = None,
    first_line: int | None = None,
    bold: bool = False,
) -> str:
    runs = "".join(
        run_text(part, bold=(part_bold or bold), italic=italic)
        for part, part_bold, italic in parse_inline_markdown(text)
    )
    return f"<w:p>{para_props(style, align, first_line)}{runs}</w:p>"


def parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    cells = parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    col_width = max(1100, 9000 // col_count)
    grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(col_count))
    body: list[str] = [
        "<w:tbl>",
        "<w:tblPr>",
        '<w:tblW w:w="0" w:type="auto"/>',
        "<w:tblBorders>",
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="A6A6A6"/>',
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="A6A6A6"/>',
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="A6A6A6"/>',
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="A6A6A6"/>',
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="D9D9D9"/>',
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="D9D9D9"/>',
        "</w:tblBorders>",
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>',
        "</w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for row_idx, row in enumerate(rows):
        body.append("<w:tr>")
        for col_idx in range(col_count):
            cell = row[col_idx] if col_idx < len(row) else ""
            shade = '<w:shd w:fill="EAF2F8"/>' if row_idx == 0 else ""
            body.append(
                "<w:tc>"
                f'<w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/>{shade}</w:tcPr>'
                + paragraph(cell, style="TableText", bold=(row_idx == 0))
                + "</w:tc>"
            )
        body.append("</w:tr>")
    body.append("</w:tbl>")
    body.append(paragraph(""))
    return "".join(body)


def svg_dimensions(path: Path) -> tuple[int, int]:
    root = ElementTree.parse(path).getroot()
    width_raw = root.attrib.get("width", "900")
    height_raw = root.attrib.get("height", "500")

    def parse_px(value: str) -> int:
        match = re.search(r"[\d.]+", value)
        return int(float(match.group(0))) if match else 900

    return parse_px(width_raw), parse_px(height_raw)


class DocxBuilder:
    def __init__(self) -> None:
        self.body_parts: list[str] = []
        self.media: list[tuple[str, Path]] = []
        self.image_rels: list[tuple[str, str]] = []
        self._next_rid = 4
        self._next_docpr = 1

    def add_paragraph(self, *args, **kwargs) -> None:
        self.body_parts.append(paragraph(*args, **kwargs))

    def add_table(self, rows: list[list[str]]) -> None:
        self.body_parts.append(table_xml(rows))

    def add_svg(self, svg_path: Path, caption: str) -> None:
        if not svg_path.exists():
            self.add_paragraph(f"[图文件缺失：{svg_path}]", style="Caption", align="center")
            return

        media_name = f"image{len(self.media) + 1}.svg"
        rid = f"rId{self._next_rid}"
        self._next_rid += 1
        docpr_id = self._next_docpr
        self._next_docpr += 1
        self.media.append((media_name, svg_path))
        self.image_rels.append((rid, media_name))

        width_px, height_px = svg_dimensions(svg_path)
        max_width_in = 6.3
        width_in = min(max_width_in, width_px / 96)
        height_in = width_in * height_px / max(width_px, 1)
        cx = int(width_in * 914400)
        cy = int(height_in * 914400)

        drawing = f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{docpr_id}" name="Figure {docpr_id}" descr="{xesc(caption)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{docpr_id}" name="{xesc(media_name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rid}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""
        self.body_parts.append(drawing)
        self.add_paragraph(caption, style="Caption", align="center")

    def document_xml(self) -> str:
        body = "".join(self.body_parts)
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""

    def document_rels_xml(self) -> str:
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>',
        ]
        for rid, media_name in self.image_rels:
            rels.append(
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{xesc(media_name)}"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels)
            + "</Relationships>"
        )

    def write(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", content_types_xml())
            docx.writestr("_rels/.rels", package_rels_xml())
            docx.writestr("docProps/core.xml", core_props_xml())
            docx.writestr("docProps/app.xml", app_props_xml())
            docx.writestr("word/document.xml", self.document_xml())
            docx.writestr("word/styles.xml", styles_xml())
            docx.writestr("word/settings.xml", settings_xml())
            docx.writestr("word/fontTable.xml", font_table_xml())
            docx.writestr("word/_rels/document.xml.rels", self.document_rels_xml())
            for media_name, path in self.media:
                docx.writestr(f"word/media/{media_name}", path.read_bytes())


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="svg" ContentType="image/svg+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def core_props_xml() -> str:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>BEA-Judge：面向生成式人工智能内容评估的偏差感知证据增强评判校准框架</dc:title>
  <dc:creator>BEA-Judge project</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex DOCX generator</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company>BEA-Judge project</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
    </w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="240" w:after="240"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimHei"/><w:b/><w:bCs/><w:sz w:val="34"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimHei"/><w:b/><w:bCs/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="180" w:after="100"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimHei"/><w:b/><w:bCs/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="180"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="19"/><w:i/><w:iCs/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Equation">
    <w:name w:val="Equation"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="SimSun"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="Table Text"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="18"/></w:rPr>
    <w:pPr><w:spacing w:before="0" w:after="0" w:line="260" w:lineRule="auto"/></w:pPr>
  </w:style>
</w:styles>
"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="420"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
</w:settings>
"""


def font_table_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="{W_NS}">
  <w:font w:name="Times New Roman"><w:family w:val="roman"/></w:font>
  <w:font w:name="SimSun"><w:family w:val="roman"/></w:font>
  <w:font w:name="SimHei"><w:family w:val="swiss"/></w:font>
  <w:font w:name="Cambria Math"><w:family w:val="roman"/></w:font>
</w:fonts>
"""


def markdown_to_docx(markdown_text: str) -> DocxBuilder:
    builder = DocxBuilder()
    lines = markdown_text.splitlines()
    i = 0
    in_references = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            caption = image_match.group(1)
            rel_path = image_match.group(2)
            builder.add_svg((PAPER_DIR / rel_path).resolve(), caption)
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            if title == "参考文献":
                in_references = True
            if level == 1:
                builder.add_paragraph(title, style="Title", align="center")
            elif level == 2:
                builder.add_paragraph(title, style="Heading1")
            else:
                builder.add_paragraph(title, style="Heading2")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1].strip()):
            rows = [parse_table_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_row(lines[i].strip()))
                i += 1
            builder.add_table(rows)
            continue

        if stripped == "\\[":
            formula_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "\\]":
                formula_lines.append(lines[i].strip())
                i += 1
            if i < len(lines) and lines[i].strip() == "\\]":
                i += 1
            builder.add_paragraph(" ".join(formula_lines), style="Equation", align="center")
            continue

        if in_references and re.match(r"^\[\d+\]", stripped):
            builder.add_paragraph(stripped, first_line=-360)
        elif re.match(r"^\d+\.\s+", stripped):
            builder.add_paragraph(stripped, first_line=360)
        else:
            builder.add_paragraph(stripped)
        i += 1

    return builder


def main() -> None:
    markdown = MANUSCRIPT.read_text(encoding="utf-8")
    builder = markdown_to_docx(markdown)
    builder.write(OUT_DOCX)
    print(OUT_DOCX)


if __name__ == "__main__":
    main()
