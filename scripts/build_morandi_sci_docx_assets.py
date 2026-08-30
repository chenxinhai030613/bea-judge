from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SRC_DOCX = ROOT / "paper" / "bea_judge_manuscript" / "bea_judge_manuscript.docx"
OUT_DIR = ROOT / "paper" / "bea_judge_morandi_sci_assets"
PNG_DIR = OUT_DIR / "png"
SVG_DIR = OUT_DIR / "svg"
PROMPT_DIR = OUT_DIR / "prompts"
REPORT_DIR = OUT_DIR / "reports"
DOCX_OUT = OUT_DIR / "bea_judge_morandi_sci.docx"


PALETTE = {
    "ink": "#343A40",
    "muted_ink": "#5F6670",
    "axis": "#3F454B",
    "tick": "#737982",
    "grid": "#E9E7E2",
    "paper": "#FFFFFF",
    "warm_panel": "#F7F5F0",
    "dusty_blue": "#7E9AAF",
    "sage": "#8AA08A",
    "clay": "#C28F85",
    "ochre": "#C6AA6A",
    "lavender": "#A69DB8",
    "slate": "#8E9AA6",
    "rose": "#B98282",
    "teal": "#7FA6A1",
    "sand": "#D4C7A1",
    "line_dark": "#56616C",
}

MORANDI_SERIES = [
    PALETTE["dusty_blue"],
    PALETTE["sage"],
    PALETTE["clay"],
    PALETTE["ochre"],
    PALETTE["lavender"],
    PALETTE["slate"],
    PALETTE["rose"],
    PALETTE["teal"],
]

TARGET_MEDIA = {
    "media/image1.png": "fig1_framework",
    "media/image53.png": "fig0_epoch_four_panel",
    "media/image54.png": "fig2_operating_points",
    "media/image56.png": "fig2b_main_external",
    "media/image57.png": "fig3_module_ablation",
    "media/image59.png": "fig4_training_ablation",
    "media/image61.png": "fig6_external_baseline",
    "media/image63.png": "fig5_risk_review",
}


@dataclass(frozen=True)
class FigureSpec:
    key: str
    stem: str
    title: str
    media: str
    size: tuple[int, int]
    caption: str
    prompt: str


FIGURES = [
    FigureSpec(
        "fig1_framework",
        "fig1_framework",
        "BEA-Judge four-module framework",
        "media/image1.png",
        (2048, 2048),
        "图1 BEA-Judge四模块框架与信息流",
        "Create a clean Morandi/Sci infographic flowchart for BEA-Judge, with four modules: Base Judge scoring, Bias awareness, Evidence factuality, Fusion calibration, plus Tie rescue and Review flag. Use a white background, low-saturation Morandi palette, Arial-style sans-serif typography, short labels, thin dark-gray arrows, no decorative shadows.",
    ),
    FigureSpec(
        "fig0_epoch_four_panel",
        "fig0_epoch_four_panel",
        "QLoRA epoch ablation",
        "media/image53.png",
        (2048, 1536),
        "表3 QLoRA训练轮数消融的图形化摘要",
        "Create a four-panel SCI line-chart figure for QLoRA epoch ablation. Panels: Accuracy, Macro-F1, ECE, Tie recall. X-axis: 0.5, 1, 2 epochs. Use Morandi colors, white background, Arial-style labels, clear error bars, no excessive gridlines, concise legends.",
    ),
    FigureSpec(
        "fig2_operating_points",
        "fig2_operating_points",
        "Main operating-point comparison",
        "media/image54.png",
        (2400, 1328),
        "图2 主要操作点测试集指标对比",
        "Create a grouped bar SCI chart comparing five BEA-Judge operating points across Accuracy, Macro-F1, ECE, and Tie recall. Use a soft Morandi palette with contrast, white background, Arial typography, clear legend, exact-looking axes, and minimal gridlines.",
    ),
    FigureSpec(
        "fig2b_main_external",
        "fig2b_main_external",
        "Main model and external baselines",
        "media/image56.png",
        (2600, 1166),
        "主模型与轻量外部基线对比",
        "Create a wide grouped bar SCI chart comparing Current BEA-Judge, QLoRA epoch2, QLoRA epoch2 plus Tie rescue, GRM, Qwen, and GLIDER. Metrics: Accuracy, Macro-F1, ECE, Tie recall. Use Morandi colors, concise labels, no 3D effects.",
    ),
    FigureSpec(
        "fig3_module_ablation",
        "fig3_module_ablation",
        "Four-module ablation",
        "media/image57.png",
        (2400, 1307),
        "图3 四模块消融结果",
        "Create a two-panel SCI grouped bar figure for BEA-Judge module ablation. Left panel: pairwise variants. Right panel: factuality variants. Metrics include Accuracy, Macro-F1, ECE, and Tie recall where applicable. Use Morandi palette, Arial typography, short labels, and clear legend.",
    ),
    FigureSpec(
        "fig4_training_ablation",
        "fig4_training_ablation",
        "QLoRA training ablation",
        "media/image59.png",
        (2400, 1200),
        "图4 QLoRA训练消融",
        "Create a two-panel SCI line-chart figure. Left panel: epoch ablation; right panel: SFT size ablation. Show Accuracy, Macro-F1, ECE, Tie recall. Use Morandi colors, small markers, clear legend, Arial labels, white background, no extra gridlines.",
    ),
    FigureSpec(
        "fig6_external_baseline",
        "fig6_external_baseline",
        "External lightweight baselines",
        "media/image61.png",
        (2400, 1265),
        "图6 外部轻量基线比较",
        "Create a grouped bar SCI chart for external lightweight baseline comparison: Current BEA-Judge, GRM, Qwen, GLIDER, QLoRA epoch2, QLoRA epoch2 plus Tie rescue. Metrics: Accuracy, Macro-F1, ECE, Tie recall. Use Morandi palette, clear legend, minimal axes.",
    ),
    FigureSpec(
        "fig5_risk_review",
        "fig5_risk_review",
        "Risk-threshold review curve",
        "media/image63.png",
        (2200, 1443),
        "图5 风险阈值复核曲线",
        "Create a clean SCI line chart for risk-threshold review curves. X-axis: manual review proportion. Y-axis: error capture rate. Two series: pairwise preference and factuality. Use Morandi blue and sage green, Arial typography, direct labels, no decorative grid.",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_mean(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("±", "+/-")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else math.nan


def parse_std(value: str | float | int) -> float | None:
    text = str(value).replace("±", "+/-")
    m = re.search(r"\+/-\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def extract_docx_tables_and_media() -> dict:
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def text_of(el: ET.Element) -> str:
        return "".join(t.text or "" for t in el.iter(f"{{{ns['w']}}}t")).strip()

    out: dict[str, object] = {"tables": [], "media": []}
    with zipfile.ZipFile(SRC_DOCX) as z:
        rels_root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        rels = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rels_root}
        doc = ET.fromstring(z.read("word/document.xml"))
        body = doc.find("w:body", ns)
        if body is None:
            return out
        last_para = ""
        for child in body:
            if child.tag == f"{{{ns['w']}}}p":
                txt = text_of(child)
                if txt:
                    last_para = txt
                for inline in list(child.iter(f"{{{ns['wp']}}}inline")) + list(child.iter(f"{{{ns['wp']}}}anchor")):
                    ext = inline.find("wp:extent", ns)
                    docpr = inline.find("wp:docPr", ns)
                    rid = None
                    for blip in inline.iter(f"{{{ns['a']}}}blip"):
                        rid = blip.attrib.get(f"{{{ns['r']}}}embed")
                    target = rels.get(rid)
                    if target in TARGET_MEDIA:
                        cx = int(ext.attrib.get("cx", "0")) if ext is not None else 0
                        cy = int(ext.attrib.get("cy", "0")) if ext is not None else 0
                        out["media"].append(
                            {
                                "target": target,
                                "figure_key": TARGET_MEDIA[target],
                                "docPr": docpr.attrib if docpr is not None else {},
                                "extent_inches": [round(cx / 914400, 3), round(cy / 914400, 3)],
                                "nearby_text": last_para,
                            }
                        )
            elif child.tag == f"{{{ns['w']}}}tbl":
                rows = []
                for tr in child.findall(".//w:tr", ns):
                    cells = [text_of(tc) for tc in tr.findall("./w:tc", ns)]
                    if any(cells):
                        rows.append(cells)
                if rows:
                    out["tables"].append({"title_before": last_para, "rows": rows})
    return out


def table_by_title(doc_info: dict, title_hint: str) -> list[list[str]]:
    for table in doc_info["tables"]:
        if title_hint in table["title_before"]:
            return table["rows"]
    return []


def rows_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if len(rows) < 2:
        return []
    headers = [h.strip() for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def row_get_case(row: dict[str, str], key: str, default: str = "") -> str:
    if key in row:
        return row[key]
    wanted = key.casefold()
    for existing, value in row.items():
        if existing.casefold() == wanted:
            return value
    return default


def load_data(doc_info: dict) -> dict:
    epoch = rows_to_dicts(table_by_title(doc_info, "表 3 QLoRA训练轮数消融"))
    operating = rows_to_dicts(table_by_title(doc_info, "表3显示"))
    main = rows_to_dicts(table_by_title(doc_info, "表4 主对比表"))
    module = rows_to_dicts(table_by_title(doc_info, "图3说明"))
    external = rows_to_dicts(table_by_title(doc_info, "外部轻量基线比较"))
    risk = rows_to_dicts(table_by_title(doc_info, "风险覆盖曲线"))
    sft_size = [
        {"setting": "25%", "Accuracy": "0.7740", "Macro-F1": "0.6832", "ECE": "0.0310", "Tie Recall": "0.4436"},
        {"setting": "50%", "Accuracy": "0.7927", "Macro-F1": "0.7012", "ECE": "0.0303", "Tie Recall": "0.4590"},
        {"setting": "100%", "Accuracy": "0.8025", "Macro-F1": "0.7128", "ECE": "0.0279", "Tie Recall": "0.4538"},
    ]
    factuality_risk = [
        {"review_rate": "0.0495", "error_capture_rate": "0.0702"},
        {"review_rate": "0.0990", "error_capture_rate": "0.1579"},
        {"review_rate": "0.2000", "error_capture_rate": "0.3509"},
        {"review_rate": "0.3010", "error_capture_rate": "0.4912"},
        {"review_rate": "0.4000", "error_capture_rate": "0.6491"},
        {"review_rate": "0.4990", "error_capture_rate": "0.7544"},
        {"review_rate": "0.7505", "error_capture_rate": "0.9649"},
        {"review_rate": "1.0000", "error_capture_rate": "1.0000"},
    ]
    return {
        "epoch": epoch,
        "operating": operating,
        "main": main,
        "module": module,
        "external": external,
        "risk": risk,
        "sft_size": sft_size,
        "factuality_risk": factuality_risk,
    }


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.strip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))


def choose_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for c in candidates:
        path = Path(c)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    color: str = PALETTE["ink"],
    anchor: str = "mm",
    bold: bool = False,
) -> None:
    draw.text(xy, text, fill=hex_to_rgb(color), font=choose_font(size, bold), anchor=anchor)


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if len(words) <= 1 and len(text) > width:
        return [text[i : i + width] for i in range(0, len(text), width)]
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else cur + " " + word
        if len(nxt) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 24,
    color: str = PALETTE["ink"],
    anchor: str = "middle",
    weight: int = 400,
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Microsoft YaHei, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" {extra}>{escape(text)}</text>'
    )


def svg_wrap(body: list[str], width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["axis"]}"/>',
            "</marker>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>',
            *body,
            "</svg>",
        ]
    )


def save_svg(path: Path, body: list[str], width: int, height: int) -> None:
    path.write_text(svg_wrap(body, width, height), encoding="utf-8")


def nice_metric(metric: str) -> str:
    return {
        "Accuracy": "Accuracy",
        "accuracy": "Accuracy",
        "Macro-F1": "Macro-F1",
        "macro-F1": "Macro-F1",
        "ECE": "ECE",
        "Tie recall": "Tie recall",
        "Tie Recall": "Tie recall",
    }.get(metric, metric)


def chart_svg_axes(
    body: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    y_label: str = "Score",
    x_label: str = "",
    y_max: float = 1.0,
    y_ticks: list[float] | None = None,
) -> None:
    ticks = y_ticks or [0, 0.25, 0.5, 0.75, 1.0]
    body.append(f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="{PALETTE["axis"]}" stroke-width="2"/>')
    body.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="{PALETTE["axis"]}" stroke-width="2"/>')
    for tick in ticks:
        ty = y + h * (1 - tick / y_max)
        body.append(f'<line x1="{x-8}" y1="{ty:.1f}" x2="{x}" y2="{ty:.1f}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
        body.append(svg_text(x - 14, ty + 6, f"{tick:.2f}", 18, PALETTE["tick"], "end"))
    body.append(svg_text(x - 58, y + h / 2, y_label, 21, PALETTE["axis"], "middle", extra='transform="rotate(-90 %.1f %.1f)"' % (x - 58, y + h / 2)))
    if x_label:
        body.append(svg_text(x + w / 2, y + h + 62, x_label, 21, PALETTE["axis"]))


def draw_axes(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    y_label: str,
    x_label: str = "",
    y_max: float = 1.0,
    y_ticks: list[float] | None = None,
) -> None:
    ticks = y_ticks or [0, 0.25, 0.5, 0.75, 1.0]
    axis = hex_to_rgb(PALETTE["axis"])
    draw.line([(x, y), (x, y + h), (x + w, y + h)], fill=axis, width=3)
    for tick in ticks:
        ty = y + h * (1 - tick / y_max)
        draw.line([(x - 10, ty), (x, ty)], fill=axis, width=2)
        draw_text(draw, (x - 18, ty), f"{tick:.2f}", 24, PALETTE["tick"], anchor="rm")
    draw_text(draw, (x, y - 34), y_label, 25, PALETTE["axis"], anchor="lm")
    if x_label:
        draw_text(draw, (x + w / 2, y + h + 78), x_label, 26, PALETTE["axis"])


def render_framework(spec: FigureSpec) -> None:
    width, height = spec.size
    img = Image.new("RGB", spec.size, hex_to_rgb(PALETTE["paper"]))
    d = ImageDraw.Draw(img)
    draw_text(d, (width / 2, 88), "BEA-Judge 四模块融合校准框架", 54, bold=True)
    draw_text(d, (width / 2, 144), "Score signals, evidence checks, bias awareness, calibrated fusion, and review routing", 28, PALETTE["muted_ink"])

    boxes = [
        (130, 375, 360, 210, PALETTE["warm_panel"], "Input sample", ["Prompt/context", "Answer A / B", "Reference/evidence"]),
        (650, 260, 420, 230, "#EEF2F1", "Module 1", ["Base Judge scoring", "Score gap", "Initial label"]),
        (650, 660, 420, 230, "#F4F0EA", "Module 2", ["Bias awareness", "Position/length", "Format/rubric risk"]),
        (1180, 660, 420, 230, "#F4ECEB", "Module 3", ["Evidence factuality", "Entity/number checks", "Support coverage"]),
        (1330, 285, 420, 250, "#EFEDF3", "Module 4", ["Fusion calibration", "Temperature scaling", "Confidence/risk"]),
        (1330, 1030, 420, 210, "#F4F0E1", "Tie rescue", ["Dev-only selection", "Accuracy-constrained", "Boundary samples"]),
        (400, 1350, 520, 210, "#EFF3F4", "Output", ["A>B / B>A / Tie", "Confidence", "Review flag"]),
    ]
    for x, y, w, h, fill, title, lines in boxes:
        d.rounded_rectangle([x, y, x + w, y + h], radius=22, fill=hex_to_rgb(fill), outline=hex_to_rgb(PALETTE["axis"]), width=4)
        draw_text(d, (x + w / 2, y + 54), title, 34, bold=True)
        for i, line in enumerate(lines):
            draw_text(d, (x + w / 2, y + 105 + i * 42), line, 25, PALETTE["muted_ink"])

    def arrow(a: tuple[int, int], b: tuple[int, int]) -> None:
        d.line([a, b], fill=hex_to_rgb(PALETTE["axis"]), width=6)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        p1 = (b[0] - 24 * math.cos(ang - 0.45), b[1] - 24 * math.sin(ang - 0.45))
        p2 = (b[0] - 24 * math.cos(ang + 0.45), b[1] - 24 * math.sin(ang + 0.45))
        d.polygon([b, p1, p2], fill=hex_to_rgb(PALETTE["axis"]))

    for a, b in [
        ((490, 480), (650, 375)),
        ((490, 480), (650, 775)),
        ((1070, 775), (1180, 775)),
        ((1070, 375), (1330, 410)),
        ((1600, 535), (1540, 1030)),
        ((1330, 1150), (920, 1455)),
        ((1330, 410), (920, 1455)),
        ((400, 1455), (250, 585)),
    ]:
        arrow(a, b)

    draw_text(d, (width / 2, 1880), "Review priority is triggered by low confidence or elevated evidence/bias risk.", 30, PALETTE["muted_ink"])
    img.save(PNG_DIR / f"{spec.stem}.png", quality=95)

    body: list[str] = [
        svg_text(width / 2, 88, "BEA-Judge 四模块融合校准框架", 54, weight=700),
        svg_text(width / 2, 144, "Score signals, evidence checks, bias awareness, calibrated fusion, and review routing", 28, PALETTE["muted_ink"]),
    ]
    for x, y, w, h, fill, title, lines in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{PALETTE["axis"]}" stroke-width="4"/>')
        body.append(svg_text(x + w / 2, y + 62, title, 34, weight=700))
        for i, line in enumerate(lines):
            body.append(svg_text(x + w / 2, y + 114 + i * 42, line, 25, PALETTE["muted_ink"]))
    for a, b in [
        ((490, 480), (650, 375)),
        ((490, 480), (650, 775)),
        ((1070, 775), (1180, 775)),
        ((1070, 375), (1330, 410)),
        ((1600, 535), (1540, 1030)),
        ((1330, 1150), (920, 1455)),
        ((1330, 410), (920, 1455)),
        ((400, 1455), (250, 585)),
    ]:
        body.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{PALETTE["axis"]}" stroke-width="6" marker-end="url(#arrow)"/>')
    body.append(svg_text(width / 2, 1880, "Review priority is triggered by low confidence or elevated evidence/bias risk.", 30, PALETTE["muted_ink"]))
    save_svg(SVG_DIR / f"{spec.stem}.svg", body, width, height)


def grouped_bar_svg(
    spec: FigureSpec,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str = "Metric value",
    panel_label: str | None = None,
) -> None:
    width, height = spec.size
    body = [svg_text(width / 2, 58, spec.title, 38, weight=700)]
    x, y, w, h = 170, 145, width - 260, height - 310
    chart_svg_axes(body, x, y, w, h, y_label)
    group_w = w / len(groups)
    bar_w = min(58, group_w / (len(series) + 1.5))
    for gi, group in enumerate(groups):
        cx = x + gi * group_w + group_w / 2
        label_lines = group.split("\n")
        for li, part in enumerate(label_lines):
            body.append(svg_text(cx, y + h + 42 + li * 26, part, 21, PALETTE["axis"]))
        for si, (name, vals, color) in enumerate(series):
            bx = cx - len(series) * bar_w / 2 + si * bar_w
            val = vals[gi]
            bh = h * val
            by = y + h - bh
            body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w * 0.78:.1f}" height="{bh:.1f}" fill="{color}"/>')
            if val >= 0.055:
                body.append(svg_text(bx + bar_w * 0.39, by - 8, f"{val:.3f}", 18, PALETTE["muted_ink"]))
    if panel_label:
        body.append(svg_text(x, 112, panel_label, 27, PALETTE["ink"], "start", 700))
    lx = x + 16
    ly = height - 70
    step = min(340, (width - x - 90) / max(1, len(series)))
    for i, (name, _, color) in enumerate(series):
        xx = lx + i * step
        body.append(f'<rect x="{xx:.1f}" y="{ly - 18:.1f}" width="28" height="18" fill="{color}"/>')
        body.append(svg_text(xx + 40, ly - 2, name, 21, PALETTE["ink"], "start"))
    save_svg(SVG_DIR / f"{spec.stem}.svg", body, width, height)
    grouped_bar_png(spec, groups, series, y_label)


def grouped_bar_png(spec: FigureSpec, groups: list[str], series: list[tuple[str, list[float], str]], y_label: str = "Metric value") -> None:
    width, height = spec.size
    img = Image.new("RGB", spec.size, hex_to_rgb(PALETTE["paper"]))
    d = ImageDraw.Draw(img)
    draw_text(d, (width / 2, 62), spec.title, 46, bold=True)
    x, y, w, h = 170, 145, width - 260, height - 310
    draw_axes(d, x, y, w, h, y_label)
    group_w = w / len(groups)
    bar_w = min(58, group_w / (len(series) + 1.5))
    for gi, group in enumerate(groups):
        cx = x + gi * group_w + group_w / 2
        for li, part in enumerate(group.split("\n")):
            draw_text(d, (cx, y + h + 48 + li * 30), part, 24, PALETTE["axis"])
        for si, (_name, vals, color) in enumerate(series):
            bx = cx - len(series) * bar_w / 2 + si * bar_w
            val = vals[gi]
            bh = h * val
            by = y + h - bh
            d.rectangle([bx, by, bx + bar_w * 0.78, y + h], fill=hex_to_rgb(color))
            if val >= 0.055:
                draw_text(d, (bx + bar_w * 0.39, by - 18), f"{val:.3f}", 19, PALETTE["muted_ink"])
    lx = x + 16
    ly = height - 76
    step = min(340, (width - x - 90) / max(1, len(series)))
    for i, (name, _vals, color) in enumerate(series):
        xx = lx + i * step
        d.rectangle([xx, ly - 24, xx + 34, ly - 6], fill=hex_to_rgb(color))
        draw_text(d, (xx + 46, ly - 15), name, 24, anchor="lm")
    img.save(PNG_DIR / f"{spec.stem}.png", quality=95)


def multi_panel_line(
    spec: FigureSpec,
    panels: list[dict],
    shared_legend: bool = False,
) -> None:
    width, height = spec.size
    body = [svg_text(width / 2, 58, spec.title, 38, weight=700)]
    img = Image.new("RGB", spec.size, hex_to_rgb(PALETTE["paper"]))
    d = ImageDraw.Draw(img)
    draw_text(d, (width / 2, 62), spec.title, 46, bold=True)
    cols = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / cols)
    panel_w = (width - 180) / cols
    panel_h = (height - 190) / rows
    colors_seen: list[tuple[str, str]] = []

    for pi, panel in enumerate(panels):
        col = pi % cols
        row = pi // cols
        px = 105 + col * panel_w
        py = 135 + row * panel_h
        cw = panel_w - 100
        ch = panel_h - 115
        title = panel["title"]
        xs = panel["xs"]
        series = panel["series"]
        y_max = panel.get("y_max", 1.0)
        y_ticks = panel.get("y_ticks", [0, 0.25, 0.5, 0.75, 1.0])
        body.append(svg_text(px + cw / 2, py - 34, f"({chr(97 + pi)}) {title}", 25, PALETTE["ink"], weight=700))
        draw_text(d, (px + cw / 2, py - 38), f"({chr(97 + pi)}) {title}", 29, bold=True)
        chart_svg_axes(body, px, py, cw, ch, panel.get("ylabel", "Value"), panel.get("xlabel", ""), y_max, y_ticks)
        draw_axes(d, int(px), int(py), int(cw), int(ch), panel.get("ylabel", "Value"), panel.get("xlabel", ""), y_max, y_ticks)
        xcoords = [px + cw * i / max(1, len(xs) - 1) for i in range(len(xs))]
        for xi, label in zip(xcoords, xs):
            body.append(svg_text(xi, py + ch + 40, str(label), 20, PALETTE["axis"]))
            draw_text(d, (xi, py + ch + 46), str(label), 23, PALETTE["axis"])
        for name, vals, color, errs in series:
            if (name, color) not in colors_seen:
                colors_seen.append((name, color))
            points = []
            for xi, val in zip(xcoords, vals):
                yy = py + ch * (1 - val / y_max)
                points.append((xi, yy))
            body.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="4"/>' % (" ".join(f"{a:.1f},{b:.1f}" for a, b in points), color))
            d.line(points, fill=hex_to_rgb(color), width=5)
            for idx, (xi, val) in enumerate(zip(xcoords, vals)):
                yy = py + ch * (1 - val / y_max)
                err = errs[idx] if errs else None
                if err:
                    y1 = py + ch * (1 - (val - err) / y_max)
                    y2 = py + ch * (1 - (val + err) / y_max)
                    body.append(f'<line x1="{xi:.1f}" y1="{y1:.1f}" x2="{xi:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2"/>')
                    body.append(f'<line x1="{xi-10:.1f}" y1="{y1:.1f}" x2="{xi+10:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="2"/>')
                    body.append(f'<line x1="{xi-10:.1f}" y1="{y2:.1f}" x2="{xi+10:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2"/>')
                    d.line([(xi, y1), (xi, y2)], fill=hex_to_rgb(color), width=3)
                    d.line([(xi - 11, y1), (xi + 11, y1)], fill=hex_to_rgb(color), width=3)
                    d.line([(xi - 11, y2), (xi + 11, y2)], fill=hex_to_rgb(color), width=3)
                body.append(f'<circle cx="{xi:.1f}" cy="{yy:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
                d.ellipse([xi - 8, yy - 8, xi + 8, yy + 8], fill=hex_to_rgb(color), outline=(255, 255, 255), width=2)
                body.append(svg_text(xi, yy - 14, f"{val:.3f}", 17, PALETTE["muted_ink"]))
                draw_text(d, (xi, yy - 19), f"{val:.3f}", 19, PALETTE["muted_ink"])
            if not shared_legend and len(series) > 1:
                lx = points[-1][0] + 12
                ly = min(max(points[-1][1] + 5, py + 18), py + ch - 8)
                body.append(svg_text(lx, ly, name, 18, color, "start", 700))
                draw_text(d, (lx, ly), name, 20, color, "lm", True)

    if shared_legend:
        lx = 140
        ly = height - 42
        step = min(360, (width - 220) / max(1, len(colors_seen)))
        for i, (name, color) in enumerate(colors_seen):
            xx = lx + i * step
            body.append(f'<line x1="{xx:.1f}" y1="{ly:.1f}" x2="{xx+36:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="5"/>')
            body.append(f'<circle cx="{xx+18:.1f}" cy="{ly:.1f}" r="7" fill="{color}"/>')
            body.append(svg_text(xx + 48, ly + 7, name, 21, PALETTE["ink"], "start"))
            d.line([(xx, ly), (xx + 36, ly)], fill=hex_to_rgb(color), width=5)
            d.ellipse([xx + 10, ly - 8, xx + 26, ly + 8], fill=hex_to_rgb(color))
            draw_text(d, (xx + 48, ly), name, 23, PALETTE["ink"], "lm")

    save_svg(SVG_DIR / f"{spec.stem}.svg", body, width, height)
    img.save(PNG_DIR / f"{spec.stem}.png", quality=95)


def panel_grouped_svg_png(spec: FigureSpec, panels: list[dict]) -> None:
    width, height = spec.size
    body = [svg_text(width / 2, 58, spec.title, 38, weight=700)]
    img = Image.new("RGB", spec.size, hex_to_rgb(PALETTE["paper"]))
    d = ImageDraw.Draw(img)
    draw_text(d, (width / 2, 62), spec.title, 46, bold=True)
    panel_w = (width - 160) / len(panels)
    legend_items: list[tuple[str, str]] = []
    for pi, panel in enumerate(panels):
        px = 105 + pi * panel_w
        py = 150
        cw = panel_w - 110
        ch = height - 330
        body.append(svg_text(px + cw / 2, py - 38, f"({chr(97+pi)}) {panel['title']}", 25, PALETTE["ink"], weight=700))
        draw_text(d, (px + cw / 2, py - 42), f"({chr(97+pi)}) {panel['title']}", 29, bold=True)
        chart_svg_axes(body, px, py, cw, ch, panel.get("ylabel", "Metric value"))
        draw_axes(d, int(px), int(py), int(cw), int(ch), panel.get("ylabel", "Metric value"))
        groups = panel["groups"]
        series = panel["series"]
        group_w = cw / len(groups)
        bar_w = min(48, group_w / (len(series) + 1.4))
        for gi, group in enumerate(groups):
            cx = px + gi * group_w + group_w / 2
            for li, line in enumerate(wrap_text(group, 9)[:3]):
                body.append(svg_text(cx, py + ch + 34 + li * 23, line, 17, PALETTE["axis"]))
                draw_text(d, (cx, py + ch + 40 + li * 25), line, 19, PALETTE["axis"])
            for si, (name, vals, color) in enumerate(series):
                if (name, color) not in legend_items:
                    legend_items.append((name, color))
                val = vals[gi]
                bx = cx - len(series) * bar_w / 2 + si * bar_w
                bh = ch * val
                by = py + ch - bh
                body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.78:.1f}" height="{bh:.1f}" fill="{color}"/>')
                d.rectangle([bx, by, bx + bar_w * 0.78, py + ch], fill=hex_to_rgb(color))
    lx = 150
    ly = height - 52
    step = min(320, (width - 230) / max(1, len(legend_items)))
    for i, (name, color) in enumerate(legend_items):
        xx = lx + i * step
        body.append(f'<rect x="{xx:.1f}" y="{ly-20:.1f}" width="30" height="20" fill="{color}"/>')
        body.append(svg_text(xx + 42, ly - 3, name, 21, PALETTE["ink"], "start"))
        d.rectangle([xx, ly - 26, xx + 36, ly - 7], fill=hex_to_rgb(color))
        draw_text(d, (xx + 48, ly - 16), name, 23, PALETTE["ink"], "lm")
    save_svg(SVG_DIR / f"{spec.stem}.svg", body, width, height)
    img.save(PNG_DIR / f"{spec.stem}.png", quality=95)


def render_figures(data: dict) -> None:
    spec_by_key = {f.key: f for f in FIGURES}
    render_framework(spec_by_key["fig1_framework"])

    epoch = data["epoch"]
    xs_epoch = [row["epoch"] for row in epoch]
    metric_names = ["accuracy", "macro-F1", "ECE", "Tie recall"]
    panels = []
    for metric, color in zip(metric_names, MORANDI_SERIES):
        vals = [parse_mean(row[metric]) for row in epoch]
        errs = [parse_std(row[metric]) or 0.0 for row in epoch]
        ymax = 0.9 if metric != "ECE" else 0.06
        ticks = [0, 0.3, 0.6, 0.9] if metric != "ECE" else [0, 0.02, 0.04, 0.06]
        panels.append({"title": nice_metric(metric), "xs": xs_epoch, "series": [(nice_metric(metric), vals, color, errs)], "y_max": ymax, "y_ticks": ticks, "xlabel": "Epoch"})
    multi_panel_line(spec_by_key["fig0_epoch_four_panel"], panels)

    operating = data["operating"] or data["main"][:3]
    op_groups = []
    for row in operating:
        label = row.get("系统/操作点") or row.get("system") or ""
        label = label.replace("QLoRA-BEA ", "QLoRA\n").replace(" epoch", "\nepoch").replace(" + Tie rescue", "\n+Tie")
        op_groups.append(label)
    op_series = []
    for metric, color in zip(["Accuracy", "Macro-F1", "ECE", "Tie Recall"], MORANDI_SERIES[:4]):
        op_series.append((nice_metric(metric), [parse_mean(row.get(metric, row.get(metric.lower(), ""))) for row in operating], color))
    grouped_bar_svg(spec_by_key["fig2_operating_points"], op_groups, op_series)

    main = data["main"]
    main_groups = [row["system"].replace("QLoRA-BEA-Judge ", "QLoRA\n").replace(" epoch2_1024", " epoch2").replace(" + Tie rescue", "\n+Tie") for row in main]
    main_series = []
    for metric, color in zip(["accuracy", "macro-F1", "ECE", "Tie recall"], MORANDI_SERIES[:4]):
        main_series.append((nice_metric(metric), [parse_mean(row[metric]) for row in main], color))
    grouped_bar_svg(spec_by_key["fig2b_main_external"], main_groups, main_series)

    module = data["module"]
    pair = [r for r in module if r.get("任务") == "pairwise"]
    fact = [r for r in module if r.get("任务") == "factuality"]
    pair_groups = [
        r["变体"].replace("Full BEA-Judge", "Full").replace("w/o ", "w/o\n").replace("Base Judge Scores", "Base scores").replace("Tie Policy", "Tie policy")
        for r in pair
    ]
    fact_groups = [r["变体"].replace("Full BEA-Judge", "Full").replace("w/o ", "w/o\n") for r in fact]
    panels_bar = [
        {
            "title": "Pairwise preference",
            "groups": pair_groups,
            "series": [
                ("Accuracy", [parse_mean(r["Accuracy"]) for r in pair], MORANDI_SERIES[0]),
                ("Macro-F1", [parse_mean(r["Macro-F1"]) for r in pair], MORANDI_SERIES[1]),
                ("ECE", [parse_mean(r["ECE"]) for r in pair], MORANDI_SERIES[3]),
                ("Tie recall", [parse_mean(r["Tie Recall"]) for r in pair], MORANDI_SERIES[2]),
            ],
        },
        {
            "title": "Factuality",
            "groups": fact_groups,
            "series": [
                ("Accuracy", [parse_mean(r["Accuracy"]) for r in fact], MORANDI_SERIES[0]),
                ("Macro-F1", [parse_mean(r["Macro-F1"]) for r in fact], MORANDI_SERIES[1]),
                ("ECE", [parse_mean(r["ECE"]) for r in fact], MORANDI_SERIES[3]),
            ],
        },
    ]
    panel_grouped_svg_png(spec_by_key["fig3_module_ablation"], panels_bar)

    sft = data["sft_size"]
    panels_train = []
    for title, rows, xs, xlabel in [
        ("Epoch ablation", epoch, [r["epoch"] for r in epoch], "Epoch"),
        ("SFT size ablation", sft, [r["setting"] for r in sft], "SFT data proportion"),
    ]:
        series = []
        for metric, color in zip(["Accuracy", "Macro-F1", "ECE", "Tie Recall"], MORANDI_SERIES[:4]):
            series.append(
                (
                    nice_metric(metric),
                    [parse_mean(row_get_case(r, metric)) for r in rows],
                    color,
                    [parse_std(row_get_case(r, metric)) or 0.0 for r in rows],
                )
            )
        panels_train.append({"title": title, "xs": xs, "series": series, "xlabel": xlabel, "y_max": 0.9, "y_ticks": [0, 0.3, 0.6, 0.9]})
    multi_panel_line(spec_by_key["fig4_training_ablation"], panels_train, shared_legend=True)

    external = data["external"] or main
    ext_groups = [
        (r.get("系统") or r.get("system") or "").replace("Current BEA-Judge", "Current\nBEA-Judge").replace("QLoRA-BEA-Judge ", "QLoRA\n").replace(" epoch2", "\nepoch2").replace(" + Tie rescue", "\n+Tie").replace(" reward model", "")
        for r in external
    ]
    ext_series = []
    for metric, color in zip(["Accuracy", "Macro-F1", "ECE", "Tie Recall"], MORANDI_SERIES[:4]):
        ext_series.append((nice_metric(metric), [parse_mean(row_get_case(r, metric)) for r in external], color))
    grouped_bar_svg(spec_by_key["fig6_external_baseline"], ext_groups, ext_series)

    pair_risk = data["risk"]
    pair_x = [parse_mean(r["review_rate"]) for r in pair_risk] + [1.0]
    pair_y = [parse_mean(r["error_capture_rate"]) for r in pair_risk] + [1.0]
    fact_x = [parse_mean(r["review_rate"]) for r in data["factuality_risk"]]
    fact_y = [parse_mean(r["error_capture_rate"]) for r in data["factuality_risk"]]
    risk_spec = spec_by_key["fig5_risk_review"]
    risk_panels = [
        {
            "title": "Risk review",
            "xs": [f"{int(x*100)}%" for x in pair_x],
            "series": [
                ("Pairwise preference", pair_y, MORANDI_SERIES[0], [0.0] * len(pair_y)),
                ("Factuality", fact_y, MORANDI_SERIES[1], [0.0] * len(fact_y)),
            ],
            "xlabel": "Manual review proportion",
            "ylabel": "Error capture rate",
            "y_max": 1.0,
        }
    ]
    render_risk_curve(risk_spec, pair_x, pair_y, fact_x, fact_y)


def render_risk_curve(spec: FigureSpec, pair_x: list[float], pair_y: list[float], fact_x: list[float], fact_y: list[float]) -> None:
    width, height = spec.size
    body = [svg_text(width / 2, 64, spec.title, 38, weight=700)]
    img = Image.new("RGB", spec.size, hex_to_rgb(PALETTE["paper"]))
    d = ImageDraw.Draw(img)
    draw_text(d, (width / 2, 66), spec.title, 46, bold=True)
    x, y, w, h = 190, 155, width - 340, height - 320
    chart_svg_axes(body, x, y, w, h, "Error capture rate", "Manual review proportion")
    draw_axes(d, x, y, w, h, "Error capture rate", "Manual review proportion")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = x + w * tick
        body.append(f'<line x1="{tx:.1f}" y1="{y+h}" x2="{tx:.1f}" y2="{y+h+8}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
        body.append(svg_text(tx, y + h + 42, f"{int(tick*100)}%", 20, PALETTE["tick"]))
        d.line([(tx, y + h), (tx, y + h + 10)], fill=hex_to_rgb(PALETTE["axis"]), width=2)
        draw_text(d, (tx, y + h + 48), f"{int(tick*100)}%", 23, PALETTE["tick"])

    for name, xs, ys, color, label_y in [
        ("Pairwise preference", pair_x, pair_y, MORANDI_SERIES[0], y + 90),
        ("Factuality", fact_x, fact_y, MORANDI_SERIES[1], y + 130),
    ]:
        pts = [(x + w * a, y + h * (1 - b)) for a, b in zip(xs, ys)]
        body.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="5"/>' % (" ".join(f"{a:.1f},{b:.1f}" for a, b in pts), color))
        d.line(pts, fill=hex_to_rgb(color), width=6)
        for px, py in pts:
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=hex_to_rgb(color), outline=(255, 255, 255), width=2)
        body.append(f'<line x1="{x+w-470}" y1="{label_y}" x2="{x+w-420}" y2="{label_y}" stroke="{color}" stroke-width="6"/>')
        body.append(f'<circle cx="{x+w-445}" cy="{label_y}" r="8" fill="{color}"/>')
        body.append(svg_text(x + w - 400, label_y + 7, name, 23, PALETTE["ink"], "start"))
        d.line([(x + w - 470, label_y), (x + w - 420, label_y)], fill=hex_to_rgb(color), width=6)
        d.ellipse([x + w - 454, label_y - 9, x + w - 436, label_y + 9], fill=hex_to_rgb(color))
        draw_text(d, (x + w - 400, label_y), name, 26, PALETTE["ink"], "lm")
    note = "At ~50% review: pairwise captures 85.11% errors; factuality captures 75.44% errors."
    body.append(svg_text(width / 2, height - 54, note, 24, PALETTE["muted_ink"]))
    draw_text(d, (width / 2, height - 58), note, 28, PALETTE["muted_ink"])
    save_svg(SVG_DIR / f"{spec.stem}.svg", body, width, height)
    img.save(PNG_DIR / f"{spec.stem}.png", quality=95)


def write_prompts(doc_info: dict) -> None:
    palette = ", ".join(f"{k} {v}" for k, v in PALETTE.items() if k in ["dusty_blue", "sage", "clay", "ochre", "lavender", "slate"])
    prompt_rows = []
    for spec in FIGURES:
        prompt = (
            "Use case: scientific-educational\n"
            "Asset type: manuscript figure PNG\n"
            f"Primary request: {spec.prompt}\n"
            "Input images: none. Do not use existing figures as visual references.\n"
            "Scene/backdrop: flat white scientific figure canvas.\n"
            "Style/medium: clean SCI manuscript infographic/chart, Morandi color system.\n"
            f"Color palette: {palette}; dark gray axes and text.\n"
            "Typography: Arial-style sans-serif; concise English/Chinese labels only where specified.\n"
            "Constraints: no 3D, no gradients, no decorative shadows, no dense background grid, no watermark, no stock illustration style.\n"
            f"Canvas intent: {spec.size[0]}x{spec.size[1]} equivalent aspect ratio.\n"
        )
        prompt_rows.append({"figure": spec.stem, "caption": spec.caption, "prompt": prompt})
    (PROMPT_DIR / "imagegen_prompts.json").write_text(json.dumps(prompt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Image Gen prompts", "", "These prompts intentionally do not reference the existing embedded figures.", ""]
    for row in prompt_rows:
        md.extend([f"## {row['figure']}", f"Caption: {row['caption']}", "", "```text", row["prompt"].rstrip(), "```", ""])
    (PROMPT_DIR / "imagegen_prompts.md").write_text("\n".join(md), encoding="utf-8")


def replace_docx_media() -> None:
    tmp_docx = OUT_DIR / "_tmp_repacked.docx"
    replacements = {spec.media: PNG_DIR / f"{spec.stem}.png" for spec in FIGURES}
    with zipfile.ZipFile(SRC_DOCX, "r") as zin, zipfile.ZipFile(tmp_docx, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("word/") and item.filename[len("word/") :] in replacements:
                data = replacements[item.filename[len("word/") :]].read_bytes()
            zout.writestr(item, data)
    if DOCX_OUT.exists():
        DOCX_OUT.unlink()
    tmp_docx.rename(DOCX_OUT)


def contact_sheet() -> Path:
    thumbs = []
    for spec in FIGURES:
        path = PNG_DIR / f"{spec.stem}.png"
        im = Image.open(path).convert("RGB")
        im.thumbnail((520, 330), Image.Resampling.LANCZOS)
        thumbs.append((spec.stem, im.copy()))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    tw, th, pad, label_h = 560, 360, 28, 38
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    for idx, (label, im) in enumerate(thumbs):
        x = pad + (idx % cols) * (tw + pad)
        y = pad + (idx // cols) * (th + label_h + pad)
        draw_text(d, (x, y + 18), label, 24, PALETTE["ink"], "lm", True)
        d.rectangle([x, y + label_h, x + tw, y + label_h + th], outline=hex_to_rgb(PALETTE["grid"]), width=2)
        sheet.paste(im, (x + (tw - im.width) // 2, y + label_h + (th - im.height) // 2))
    out = REPORT_DIR / "contact_sheet.png"
    sheet.save(out, quality=95)
    return out


def verify(doc_info: dict, source_hash_before: str) -> dict:
    source_hash_after = sha256(SRC_DOCX)
    png_files = sorted(p.name for p in PNG_DIR.glob("*.png"))
    svg_files = sorted(p.name for p in SVG_DIR.glob("*.svg"))
    docx_ok = False
    replaced = {}
    if DOCX_OUT.exists():
        try:
            with zipfile.ZipFile(DOCX_OUT) as z:
                docx_ok = "word/document.xml" in z.namelist()
                for spec in FIGURES:
                    name = "word/" + spec.media
                    if name in z.namelist():
                        replaced[spec.media] = {
                            "bytes": len(z.read(name)),
                            "sha256": hashlib.sha256(z.read(name)).hexdigest(),
                        }
        except zipfile.BadZipFile:
            docx_ok = False
    checks = {
        "source_docx": str(SRC_DOCX),
        "source_unchanged": source_hash_before == source_hash_after,
        "output_docx": str(DOCX_OUT),
        "output_docx_exists": DOCX_OUT.exists(),
        "output_docx_unzips": docx_ok,
        "png_count": len(png_files),
        "svg_count": len(svg_files),
        "png_files": png_files,
        "svg_files": svg_files,
        "target_media_count": len(FIGURES),
        "docx_replaced_media": replaced,
        "media_slots": doc_info["media"],
        "note": "PNG/SVG figures are rebuilt from document text/tables and module semantics; old embedded images are used only to identify media slots and dimensions.",
    }
    (REPORT_DIR / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def main() -> None:
    for path in [OUT_DIR, PNG_DIR, SVG_DIR, PROMPT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(SRC_DOCX)
    doc_info = extract_docx_tables_and_media()
    (REPORT_DIR / "docx_extracted_metadata.json").write_text(json.dumps(doc_info, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prompts(doc_info)
    data = load_data(doc_info)
    (REPORT_DIR / "figure_data_used.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    render_figures(data)
    replace_docx_media()
    sheet = contact_sheet()
    checks = verify(doc_info, source_hash_before)
    manifest = {
        "output_dir": str(OUT_DIR),
        "docx": str(DOCX_OUT),
        "contact_sheet": str(sheet),
        "checks": checks,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
