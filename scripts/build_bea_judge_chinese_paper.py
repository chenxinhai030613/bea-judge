from __future__ import annotations

import csv
import html
import json
import math
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "bea_judge_manuscript"
FIG_DIR = OUT / "figures"
VSDX_DIR = OUT / "vsdx"
FORMAL = ROOT / "datasets" / "model_outputs" / "sci_tables_v2_20260521_110114"
EXT = ROOT / "datasets" / "model_outputs" / "sci_tables_extended_20260522"
QLORA = ROOT / "datasets" / "model_outputs"


COLORS = {
    "blue": "#356AA0",
    "orange": "#D9872B",
    "green": "#4D8C57",
    "red": "#B94A48",
    "purple": "#7A5A9E",
    "teal": "#3D8C8A",
    "gray": "#6B7280",
    "light": "#F5F7FA",
    "grid": "#D7DCE2",
    "text": "#1F2937",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fnum(value: str | float | int | None, digits: int = 4) -> str:
    if value in {None, ""}:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mean_std(text: str) -> tuple[float, float]:
    parts = text.replace("+/-", "±").split("±")
    return float(parts[0].strip()), float(parts[1].strip()) if len(parts) > 1 else 0.0


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def wrap(text: str, width: int = 22) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False) or [""]


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#333", width: float = 1.4) -> None:
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{esc(color)}" stroke-width="{width:.2f}" />'
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: str = "#333", width: float = 1.4) -> None:
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{esc(color)}" stroke-width="{width:.2f}" marker-end="url(#arrow)" />'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "white",
        stroke: str = "#333",
        radius: float = 4,
        width: float = 1.2,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius:.2f}" '
            f'fill="{esc(fill)}" stroke="{esc(stroke)}" stroke-width="{width:.2f}" />'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 14,
        anchor: str = "middle",
        weight: str = "normal",
        color: str = COLORS["text"],
    ) -> None:
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{esc(color)}">{esc(text)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: Iterable[str],
        size: int = 13,
        anchor: str = "middle",
        weight: str = "normal",
        color: str = COLORS["text"],
        line_height: float = 1.25,
    ) -> None:
        for i, line in enumerate(lines):
            self.text(x, y + i * size * line_height, line, size=size, anchor=anchor, weight=weight, color=color)

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "white") -> None:
        self.items.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{esc(fill)}" stroke="{esc(stroke)}" stroke-width="1" />'
        )

    def polyline(self, points: Sequence[tuple[float, float]], color: str, width: float = 2.0) -> None:
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(
            f'<polyline points="{coords}" fill="none" stroke="{esc(color)}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" />'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
                f'viewBox="0 0 {self.width} {self.height}">',
                "<defs>",
                '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
                '<path d="M0,0 L0,6 L9,3 z" fill="#4B5563" />',
                "</marker>",
                "</defs>",
                f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white" />',
                *self.items,
                "</svg>",
            ]
        )
        path.write_text(content, encoding="utf-8", newline="\n")


def axis(svg: SVG, x: float, y: float, w: float, h: float, y_max: float, y_label: str, x_label: str = "") -> None:
    svg.line(x, y, x, y + h, COLORS["text"], 1.2)
    svg.line(x, y + h, x + w, y + h, COLORS["text"], 1.2)
    for i in range(6):
        val = y_max * i / 5
        yy = y + h - (val / y_max) * h
        svg.line(x - 4, yy, x + w, yy, COLORS["grid"], 0.8)
        svg.text(x - 10, yy + 4, f"{val:.1f}", size=10, anchor="end", color=COLORS["gray"])
    svg.text(x + w / 2, y + h + 45, x_label, size=13)
    svg.text(x - 54, y + h / 2, y_label, size=13, anchor="middle")


def fig1_pipeline() -> None:
    svg = SVG(1200, 430)
    svg.text(600, 38, "图1  BEA-Judge四模块框架与信息流", size=24, weight="bold")
    boxes = [
        ("BEA-Judge-10K v2", "多源偏好、偏差与事实性样本\n许可审计与split门禁", COLORS["blue"]),
        ("基础 Judge 评分", "M-Prometheus-3B\n分数、标签、margin、swap诊断", COLORS["teal"]),
        ("偏差感知模块", "位置、长度、格式、rubric\n源域风险与复核原因", COLORS["orange"]),
        ("证据增强事实性", "上下文/参考支持度\n数值、日期、实体、否定错配", COLORS["green"]),
        ("融合校准与置信度", "softmax头、温度缩放\nTie策略、risk_score、review_flag", COLORS["purple"]),
    ]
    x0, y, w, h, gap = 42, 115, 190, 132, 38
    for i, (title, body, color) in enumerate(boxes):
        x = x0 + i * (w + gap)
        svg.rect(x, y, w, h, fill=color + "22", stroke=color, radius=10, width=2)
        svg.text(x + w / 2, y + 34, title, size=16, weight="bold")
        svg.multiline(x + w / 2, y + 66, body.split("\n"), size=12)
        if i < len(boxes) - 1:
            svg.arrow(x + w + 6, y + h / 2, x + w + gap - 8, y + h / 2, COLORS["gray"], 2)
    svg.rect(333, 315, 535, 58, fill=COLORS["light"], stroke=COLORS["gray"], radius=8)
    svg.text(600, 340, "输出：predicted_label、final_score、confidence、risk_score、review_flag、review_reason", size=15)
    svg.text(600, 362, "实验原则：所有温度、阈值与Tie策略仅在dev上选择，test只报告一次", size=13, color=COLORS["gray"])
    svg.save(FIG_DIR / "fig1_pipeline.svg")
    make_vsdx(
        VSDX_DIR / "fig1_pipeline.vsdx",
        "BEA-Judge四模块框架与信息流",
        boxes=[(title, body, i) for i, (title, body, _color) in enumerate(boxes)],
    )


def fig2_main_results() -> None:
    data = [
        ("Current", 0.7512, 0.6730, 0.0558, 0.5231),
        ("QLoRA-e1\nAccuracy", 0.8025, 0.7128, 0.0279, 0.4538),
        ("QLoRA-e1\nTie-sensitive", 0.7582, 0.7169, 0.0229, 0.7667),
        ("QLoRA-e2", 0.8297, 0.7348, 0.0278, 0.4256),
        ("QLoRA-e2\nTie rescue", 0.8297, 0.7441, 0.0283, 0.4795),
    ]
    metrics = [("Accuracy", 1), ("Macro-F1", 2), ("ECE", 3), ("Tie Recall", 4)]
    svg = SVG(1120, 620)
    svg.text(560, 36, "图2  主要操作点测试集指标对比（三seed mean）", size=23, weight="bold")
    x, y, w, h = 82, 78, 930, 415
    axis(svg, x, y, w, h, 0.9, "指标值")
    group_w = w / len(metrics)
    bar_w = group_w / 7
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for mi, (metric, idx) in enumerate(metrics):
        base_x = x + mi * group_w + 22
        for si, row in enumerate(data):
            val = row[idx]
            bh = val / 0.9 * h
            bx = base_x + si * bar_w
            by = y + h - bh
            svg.rect(bx, by, bar_w * 0.82, bh, fill=colors[si], stroke="white", radius=0, width=0.5)
            svg.text(bx + bar_w * 0.41, by - 5, f"{val:.3f}", size=9)
        svg.text(base_x + 2.5 * bar_w, y + h + 24, metric, size=13)
    lx, ly = 135, 535
    for i, row in enumerate(data):
        svg.rect(lx + i * 190, ly, 18, 18, fill=colors[i], stroke="white", radius=2)
        svg.multiline(lx + i * 190 + 24, ly + 14, row[0].split("\n"), size=11, anchor="start")
    svg.save(FIG_DIR / "fig2_main_operating_points.svg")
    make_vsdx(VSDX_DIR / "fig2_main_operating_points.vsdx", "主要操作点测试集指标对比")


def fig3_ablation() -> None:
    variants = [
        ("Full", 0.8025, 0.7128, 0.4538, 0.7649, 0.7405),
        ("w/o Bias", 0.7809, 0.7143, 0.6000, 0.7649, 0.7405),
        ("w/o Evidence", 0.8025, 0.7128, 0.4538, 0.3567, 0.2629),
        ("w/o Base", 0.5626, 0.5560, 0.9974, math.nan, math.nan),
        ("w/o Tie Policy", 0.8288, 0.7187, 0.3359, math.nan, math.nan),
    ]
    svg = SVG(1120, 610)
    svg.text(560, 38, "图3  四模块消融：成对偏好与事实性任务", size=23, weight="bold")
    x, y, w, h = 85, 78, 430, 390
    axis(svg, x, y, w, h, 1.0, "Pairwise")
    x2 = 610
    axis(svg, x2, y, w, h, 1.0, "Factuality")
    bar_w = 28
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for i, row in enumerate(variants):
        bx = x + 32 + i * 77
        for j, val in enumerate(row[1:4]):
            bh = val * h
            svg.rect(bx + j * bar_w, y + h - bh, bar_w - 3, bh, fill=colors[j], stroke="white", radius=0, width=0.5)
        svg.multiline(bx + 25, y + h + 25, row[0].split(" "), size=10)
        if not math.isnan(row[3]):
            bx2 = x2 + 50 + i * 100
            for j, val in enumerate(row[3:5]):
                bh = val * h
                svg.rect(bx2 + j * 34, y + h - bh, 30, bh, fill=colors[j], stroke="white", radius=0, width=0.5)
            svg.multiline(bx2 + 32, y + h + 25, row[0].split(" "), size=10)
    for i, (label, color) in enumerate([("Accuracy", colors[0]), ("Macro-F1", colors[1]), ("Tie Recall", colors[2])]):
        svg.rect(230 + i * 140, 525, 18, 18, fill=color, stroke="white", radius=2)
        svg.text(255 + i * 140, 539, label, size=13, anchor="start")
    svg.save(FIG_DIR / "fig3_four_module_ablation.svg")
    make_vsdx(VSDX_DIR / "fig3_four_module_ablation.vsdx", "四模块消融结果")


def fig4_epoch_sft() -> None:
    epoch = [
        ("0.5", 0.7873, 0.6943, 0.0318, 0.4385),
        ("1", 0.8025, 0.7128, 0.0279, 0.4538),
        ("2", 0.8297, 0.7348, 0.0278, 0.4256),
    ]
    sft = [
        ("25%", 0.7740, 0.6832, 0.0310, 0.4436),
        ("50%", 0.7927, 0.7012, 0.0303, 0.4590),
        ("100%", 0.8025, 0.7128, 0.0279, 0.4538),
    ]
    svg = SVG(1120, 560)
    svg.text(560, 38, "图4  QLoRA训练轮数与SFT规模消融", size=23, weight="bold")
    draw_line_panel(svg, 80, 82, 445, 340, epoch, "Epoch数", "epoch ablation")
    draw_line_panel(svg, 610, 82, 445, 340, sft, "SFT数据比例", "SFT size ablation")
    labels = [("Accuracy", COLORS["blue"]), ("Macro-F1", COLORS["orange"]), ("ECE", COLORS["green"]), ("Tie Recall", COLORS["purple"])]
    for i, (label, color) in enumerate(labels):
        svg.circle(278 + i * 150, 482, 7, color)
        svg.text(294 + i * 150, 486, label, size=13, anchor="start")
    svg.save(FIG_DIR / "fig4_training_ablations.svg")
    make_vsdx(VSDX_DIR / "fig4_training_ablations.vsdx", "QLoRA训练消融")


def draw_line_panel(svg: SVG, x: float, y: float, w: float, h: float, data: list[tuple[str, float, float, float, float]], xlabel: str, title: str) -> None:
    axis(svg, x, y, w, h, 0.9, "指标值", xlabel)
    svg.text(x + w / 2, y - 20, title, size=16, weight="bold")
    series = [
        (1, COLORS["blue"]),
        (2, COLORS["orange"]),
        (3, COLORS["green"]),
        (4, COLORS["purple"]),
    ]
    for idx, color in series:
        pts = []
        for i, row in enumerate(data):
            px = x + 50 + i * ((w - 100) / max(1, len(data) - 1))
            py = y + h - (row[idx] / 0.9) * h
            pts.append((px, py))
        svg.polyline(pts, color, 2.3)
        for px, py in pts:
            svg.circle(px, py, 5.5, color)
    for i, row in enumerate(data):
        px = x + 50 + i * ((w - 100) / max(1, len(data) - 1))
        svg.text(px, y + h + 24, row[0], size=12)


def fig5_risk_coverage() -> None:
    pair = [
        (0.0503, 0.1412),
        (0.0997, 0.2634),
        (0.2004, 0.5000),
        (0.3001, 0.6565),
        (0.3998, 0.7405),
        (0.4995, 0.8511),
        (0.7502, 1.0000),
        (1.0000, 1.0000),
    ]
    fact = [
        (0.0495, 0.0702),
        (0.0990, 0.1579),
        (0.2000, 0.3509),
        (0.3010, 0.4912),
        (0.4000, 0.6491),
        (0.4990, 0.7544),
        (0.7505, 0.9649),
        (1.0000, 1.0000),
    ]
    svg = SVG(900, 590)
    svg.text(450, 40, "图5  风险阈值复核曲线", size=23, weight="bold")
    x, y, w, h = 95, 85, 700, 380
    axis(svg, x, y, w, h, 1.0, "错误捕获率", "人工复核比例")
    for data, color, label, yy in [(pair, COLORS["blue"], "成对偏好", 505), (fact, COLORS["green"], "事实性", 530)]:
        pts = [(x + a * w, y + h - b * h) for a, b in data]
        svg.polyline(pts, color, 2.5)
        for px, py in pts:
            svg.circle(px, py, 5, color)
        svg.circle(310, yy, 7, color)
        svg.text(328, yy + 4, label, size=13, anchor="start")
    svg.text(450, 568, "复核约50%样本时：成对头捕获85.11%错误；事实性头捕获75.44%错误", size=13)
    svg.save(FIG_DIR / "fig5_risk_coverage.svg")
    make_vsdx(VSDX_DIR / "fig5_risk_coverage.vsdx", "风险阈值复核曲线")


def fig6_external_baseline() -> None:
    rows = [
        ("Current\nBEA-Judge", 0.7512, 0.6730, 0.0558, 0.5231),
        ("GRM-3B", 0.7312, 0.6584, 0.1759, 0.5000),
        ("Qwen2.5-3B", 0.5736, 0.4160, 0.3479, 0.0308),
        ("GLIDER-4B", 0.5043, 0.4273, 0.0353, 0.3692),
        ("QLoRA-BEA\nepoch2", 0.8297, 0.7348, 0.0278, 0.4256),
        ("QLoRA-BEA\ne2+Tie", 0.8297, 0.7441, 0.0283, 0.4795),
    ]
    svg = SVG(1120, 590)
    svg.text(560, 38, "图6  外部轻量基线与QLoRA-BEA-Judge比较", size=23, weight="bold")
    x, y, w, h = 80, 85, 920, 360
    axis(svg, x, y, w, h, 0.9, "指标值")
    group_w = w / 4
    bar_w = group_w / 8
    metrics = [("Accuracy", 1), ("Macro-F1", 2), ("ECE", 3), ("Tie Recall", 4)]
    colors = [COLORS["gray"], COLORS["red"], COLORS["orange"], COLORS["teal"], COLORS["blue"], COLORS["purple"]]
    for mi, (metric, idx) in enumerate(metrics):
        base_x = x + mi * group_w + 18
        for si, row in enumerate(rows):
            val = row[idx]
            bh = val / 0.9 * h
            bx = base_x + si * bar_w
            svg.rect(bx, y + h - bh, bar_w * 0.82, bh, fill=colors[si], stroke="white", radius=0, width=0.5)
        svg.text(base_x + 3 * bar_w, y + h + 24, metric, size=13)
    for i, row in enumerate(rows):
        lx = 72 + (i % 3) * 350
        ly = 500 + (i // 3) * 42
        svg.rect(lx, ly, 18, 18, fill=colors[i], stroke="white", radius=2)
        svg.multiline(lx + 26, ly + 14, row[0].split("\n"), size=11, anchor="start")
    svg.save(FIG_DIR / "fig6_external_baselines.svg")
    make_vsdx(VSDX_DIR / "fig6_external_baselines.vsdx", "外部轻量基线比较")


def minimal_vsdx_xml(title: str, boxes: list[tuple[str, str, int]] | None = None) -> dict[str, str]:
    boxes = boxes or []
    shapes = []
    shape_id = 1
    for title_text, body_text, idx in boxes:
        x = 1.1 + idx * 1.8
        y = 5.0
        text = esc(title_text + "\n" + body_text)
        shapes.append(
            f"""
            <Shape ID="{shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
              <XForm><PinX>{x:.2f}</PinX><PinY>{y:.2f}</PinY><Width>1.55</Width><Height>1.05</Height></XForm>
              <Text>{text}</Text>
            </Shape>
            """
        )
        shape_id += 1
    if not shapes:
        shapes.append(
            f"""
            <Shape ID="1" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
              <XForm><PinX>4.25</PinX><PinY>5.00</PinY><Width>5.50</Width><Height>1.20</Height></XForm>
              <Text>{esc(title)}</Text>
            </Shape>
            """
        )
    page = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PageContents xmlns="http://schemas.microsoft.com/office/visio/2012/main">
  <Shapes>
    {''.join(shapes)}
  </Shapes>
</PageContents>
"""
    return {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/>
  <Override PartName="/visio/pages/pages.xml" ContentType="application/vnd.ms-visio.pages+xml"/>
  <Override PartName="/visio/pages/page1.xml" ContentType="application/vnd.ms-visio.page+xml"/>
</Types>
""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/document" Target="visio/document.xml"/>
</Relationships>
""",
        "visio/document.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<VisioDocument xmlns="http://schemas.microsoft.com/office/visio/2012/main">
  <DocumentSettings/>
  <Colors/>
  <FaceNames/>
  <StyleSheets/>
  <DocumentSheet/>
  <Pages>
    <Page ID="0" NameU="Page-1" Name="{esc(title)}"><Rel r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></Page>
  </Pages>
</VisioDocument>
""",
        "visio/_rels/document.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/pages" Target="pages/pages.xml"/>
</Relationships>
""",
        "visio/pages/pages.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Pages xmlns="http://schemas.microsoft.com/office/visio/2012/main">
  <Page ID="0" Name="{esc(title)}" NameU="Page-1"><Rel r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></Page>
</Pages>
""",
        "visio/pages/_rels/pages.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/page" Target="page1.xml"/>
</Relationships>
""",
        "visio/pages/page1.xml": page,
    }


def make_vsdx(path: Path, title: str, boxes: list[tuple[str, str, int]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = minimal_vsdx_xml(title, boxes)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)


def source_table() -> str:
    rows = read_csv(FORMAL / "source_provenance_table.csv")
    return md_table(
        ["来源", "许可", "纳入样本", "可再分发", "训练纳入"],
        [
            [
                row["source"],
                row["license"],
                row["accepted_records"],
                "是" if row["redistribution_allowed"] == "True" else "否",
                "是" if row["admission_allowed"] == "True" else "否",
            ]
            for row in rows
        ],
    )


def main_result_table() -> str:
    rows = [r for r in read_csv(FORMAL / "main_results_table.csv") if r["split"] == "test"]
    return md_table(
        ["任务头", "n", "Accuracy", "Macro-F1", "ECE", "Brier", "Tie Recall", "Review Rate"],
        [
            [
                "成对偏好" if row["head"] == "pairwise" else "事实性",
                row["n"],
                fnum(row["accuracy"]),
                fnum(row["macro_f1"]),
                fnum(row["ece"]),
                fnum(row["brier"]),
                fnum(row["tie_recall"]),
                fnum(row["review_rate"]),
            ]
            for row in rows
        ],
    )


def qlora_table() -> str:
    return md_table(
        ["系统/操作点", "Accuracy", "Macro-F1", "ECE", "Tie Recall"],
        [
            ["Current BEA-Judge", "0.7512", "0.6730", "0.0558", "0.5231"],
            ["QLoRA-BEA epoch1 accuracy-oriented", "0.8025±0.0034", "0.7128±0.0063", "0.0279±0.0046", "0.4538±0.0400"],
            ["QLoRA-BEA epoch1 tie-sensitive", "0.7582±0.0022", "0.7169±0.0006", "0.0229±0.0046", "0.7667±0.0177"],
            ["QLoRA-BEA epoch2", "0.8297±0.0031", "0.7348±0.0062", "0.0278±0.0031", "0.4256±0.0270"],
            ["QLoRA-BEA epoch2 + Tie rescue", "0.8297±0.0052", "0.7441±0.0093", "0.0283±0.0027", "0.4795±0.0489"],
        ],
    )


def ablation_table() -> str:
    rows = [
        ["Full BEA-Judge", "pairwise", "0.8025±0.0034", "0.7128±0.0063", "0.0279±0.0046", "0.4538±0.0400"],
        ["w/o Bias Module", "pairwise", "0.7809±0.0255", "0.7143±0.0108", "0.0348±0.0146", "0.6000±0.1774"],
        ["w/o Evidence Module", "pairwise", "0.8025±0.0034", "0.7128±0.0063", "0.0279±0.0047", "0.4538±0.0400"],
        ["w/o Base Judge Scores", "pairwise", "0.5626±0.0022", "0.5560±0.0075", "0.1783±0.0332", "0.9974±0.0044"],
        ["w/o Tie Policy", "pairwise", "0.8288±0.0014", "0.7187±0.0063", "0.0197±0.0058", "0.3359±0.0193"],
        ["Full BEA-Judge", "factuality", "0.7649", "0.7405", "0.0377", "--"],
        ["w/o Evidence Module", "factuality", "0.3567", "0.2629", "0.6274", "--"],
    ]
    return md_table(["变体", "任务", "Accuracy", "Macro-F1", "ECE", "Tie Recall"], rows)


def external_table() -> str:
    return md_table(
        ["系统", "Accuracy", "Macro-F1", "ECE", "Tie Recall"],
        [
            ["Current BEA-Judge", "0.7512", "0.6730", "0.0558", "0.5231"],
            ["GRM-Llama3.2-3B reward model", "0.7312", "0.6584", "0.1759", "0.5000"],
            ["Qwen2.5-3B-Instruct", "0.5736", "0.4160", "0.3479", "0.0308"],
            ["GLIDER", "0.5043", "0.4273", "0.0353", "0.3692"],
            ["QLoRA-BEA-Judge epoch2", "0.8297±0.0031", "0.7348±0.0062", "0.0278±0.0031", "0.4256±0.0270"],
            ["QLoRA-BEA-Judge epoch2 + Tie rescue", "0.8297±0.0052", "0.7441±0.0093", "0.0283±0.0027", "0.4795±0.0489"],
        ],
    )


def references() -> str:
    return textwrap.dedent(
        """
        [1] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS Datasets and Benchmarks, 2023.
        [2] S. Kim et al., "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models," ICLR, 2024.
        [3] S. Kim et al., "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models," EMNLP, 2024.
        [4] J. Pombal et al., "M-Prometheus: A Suite of Open Multilingual LLM Judges," arXiv:2504.04953, 2025.
        [5] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017.
        [6] J. C. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," Advances in Large Margin Classifiers, MIT Press, 1999.
        [7] C. Niu et al., "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," arXiv:2401.00396, 2024.
        [8] Z. Wang et al., "HelpSteer2: Open-source Dataset for Training Top-performing Reward Models," NeurIPS Datasets and Benchmarks, 2024.
        [9] A. Köpf et al., "OpenAssistant Conversations: Democratizing Large Language Model Alignment," NeurIPS Datasets and Benchmarks, 2023.
        [10] J. Park et al., "OffsetBias: Leveraging Debiased Data for Tuning Evaluators," arXiv:2407.06551, 2024.
        """
    ).strip()


def clean_block(text: str) -> str:
    lines = text.strip().splitlines()
    cleaned = [line[8:] if line.startswith("        ") else line for line in lines]
    return "\n".join(cleaned) + "\n"


def make_paper_md() -> str:
    return clean_block(
        f"""
        # BEA-Judge：面向生成式人工智能内容评估的偏差感知证据增强评判校准框架

        作者信息：待补充

        ## 摘要

        生成式人工智能内容评估正在从静态基准测试转向可解释、可校准且可复核的评判系统。现有 LLM-as-a-Judge 方法能够降低人工评估成本，但在成对偏好排序、事实性判断和偏差控制上仍面临三类问题：评判器对位置、长度和格式等表层因素敏感；事实性错误常以局部证据缺失、数值或实体错配等细粒度形式出现；单次模型输出缺乏面向生产评审的置信度校准与复核机制。本文提出 BEA-Judge，一个建立在真实 M-Prometheus-3B 输出之上的偏差感知证据增强评判校准框架。该框架以“基础 Judge 评分模块、偏差感知模块、证据增强事实性模块、融合校准与置信度输出模块”为核心，将基座评判器分数、文本差异、偏差风险、证据支持特征和数据源元信息输入任务特定 softmax 头，并在开发集上进行温度缩放、Tie 策略选择和风险阈值选择。我们构建 BEA-Judge-10K v2，共 10,200 条样本，覆盖开放问答、成对偏差评估和 RAG 事实性判断。冻结测试结果显示，成对偏好头达到 0.7512 accuracy、0.6730 macro-F1 和 0.0558 ECE；事实性头达到 0.7649 accuracy、0.7405 macro-F1 和 0.0377 ECE。三 seed QLoRA 实验进一步表明，epoch2 QLoRA-BEA-Judge 达到 0.8297 accuracy、0.7348 macro-F1 和 0.0278 ECE；加入 dev-only Tie rescue 后 macro-F1 提升至 0.7441，Tie recall 提升至 0.4795。消融实验显示，证据增强是事实性可靠性的主要贡献，在 QLoRA 四模块回放中去除证据模块使事实性 macro-F1 降至 0.2629；偏差模块更适合作为风险识别与复核优先级机制，而非单纯追求总体准确率提升。

        **关键词：** 生成式人工智能评估；LLM-as-a-Judge；偏差感知；事实性验证；概率校准；风险复核；QLoRA

        ## 1 引言

        大语言模型输出的开放性使传统基于精确答案匹配的评估范式难以覆盖真实应用中的质量、偏好和事实性要求。LLM-as-a-Judge 通过将强模型或专用评判模型作为评审器，为开放式对话和响应排序提供了可扩展方案[1-4]。然而，近期研究也表明，评判器容易受到位置偏差、长度偏差、格式偏差和准则敏感性的影响；这些偏差会降低评估结论的可解释性，尤其会影响需要人工复核或质量门禁的场景。

        本文关注两类高频评估任务。第一类是成对偏好评估，即给定提示、上下文或评分准则，判断候选回答 A、B 的相对质量或是否应判为 Tie。第二类是事实性评估，即判断回答是否被上下文或参考证据支持。成对任务要求模型处理偏好边界、Tie 标签和候选顺序敏感性；事实性任务则要求系统识别实体、数值、日期、否定和比较关系等局部证据错配。单独依赖基座评判器输出，会把这些异质误差压缩成一个未校准标签，不利于解释和复核。

        参考 CodeUltraFeedback 的论文组织方式，本文采用“数据与问题定义-数据分析-模型/对齐方法-实验评估-讨论”的逻辑展开。与直接训练大型评判器不同，BEA-Judge 将 M-Prometheus-3B 的真实评判输出作为基座信号，再加入文本结构、偏差风险、证据支持和数据源特征，训练轻量级任务特定 softmax 头。随后，系统在开发集上选择温度缩放、数据集级温度策略、Tie 决策策略和低置信复核阈值。该设计将自动评判拆分为“基础判断、风险诊断、概率校准和复核控制”四个可审计环节，便于在科研报告和工程部署中追踪误差来源。

        本文贡献如下：

        1. 提出一种面向生成式内容评估的偏差感知证据增强校准框架，将基座 LLM 评判输出、偏差风险特征和证据事实性特征统一到任务特定概率模型中。
        2. 构建并审计 BEA-Judge-10K v2，覆盖开放问答、成对偏差和 RAG 事实性三类任务，并明确训练、开发和测试划分及数据许可状态。
        3. 在冻结测试集与三 seed QLoRA 协议上报告主结果、消融实验、外部轻量基线、校准与 Tie 策略分析，给出可复现的实验门禁。
        4. 明确方法边界：BEA-Judge 不是单纯的新评判大模型，也不声称解决原子声明级事实核验；其主要作用是提升评判输出的校准性、证据敏感性和复核可控性。

        ![图1 BEA-Judge四模块框架与信息流](figures/fig1_pipeline.svg)

        ## 2 相关工作

        ### 2.1 LLM-as-a-Judge 与开放评判模型

        MT-Bench 和 Chatbot Arena 推动了使用 LLM 近似人类偏好评估的研究，证明强模型在一定条件下可作为可扩展评审器[1]。Prometheus 系列进一步强调开放评判模型、评分准则和参考材料的重要性[2,3]。M-Prometheus 将评判能力扩展到多语言直接评分和成对比较场景[4]，因此适合作为本文中文与英文混合实验的基座评判器。与这些工作不同，本文不试图仅以更大评判器替代人工，而是在评判器输出之上建立可校准、可诊断的轻量融合层。

        ### 2.2 偏差鲁棒性与事实性评估

        LLM 评判器已知存在位置、长度、格式和准则敏感性等偏差[1,10]。OffsetBias 从数据层面构建去偏训练样本，而本文将偏差作为风险诊断信号，用于校准和复核优先级。事实性方面，RAGTruth 提供了检索增强生成中幻觉和证据不一致的细粒度语料[7]。本文借鉴其证据敏感问题设置，但采用确定性证据特征和任务特定概率头，而非直接进行原子声明级判定。

        ### 2.3 概率校准与风险复核

        现代神经模型经常出现置信度与正确率不匹配的问题，温度缩放是常用后处理方法[5]。Platt scaling 及其后续校准方法为概率输出校正提供了经典基线[6]。本文使用开发集选择温度缩放、数据集级温度和复核阈值，并额外报告多种校准方法在成对偏好头上的对比。

        ## 3 数据集构建与分析

        ### 3.1 数据来源与任务组成

        BEA-Judge-10K v2 由 10,200 条样本组成，包含三类任务：开放问答 4,000 条、成对偏差评估 2,700 条、RAG 事实性评估 3,500 条。数据来源包括 HelpSteer2、OpenAssistant Conversations、OffsetBias、RAGTruth 以及项目内中文专业标注数据。训练、开发和测试划分分别为 7,084、1,578 和 1,538 条；语言分布为英文 9,148 条、中文 1,052 条。

        表1 数据来源、许可与纳入状态。

        {source_table()}

        ### 3.2 质量控制

        数据构建执行样本数、字段类型、枚举值、重复 ID、重复内容和跨 split 泄漏检查。正式结果门禁要求样本数位于 9,500 到 10,200 区间，成对基座评判覆盖率为 6,946/6,946，启发式 fallback 行数为 0，未解析失败为 0。实验索引显示所有正式门禁均通过，包括基座分数覆盖、修复后解析覆盖、无启发式正式结果、消融变体存在、偏差预测覆盖率为 1.0、证据 profile 数量完整以及事实性 ECE 不超过 0.04。

        ## 4 方法：四模块 BEA-Judge 框架

        ### 4.1 问题形式化

        对样本 \\(x_i\\)，成对偏好头的标签空间为 \\(\\mathcal{{Y}}_p=\\{{A>B,B>A,Tie\\}}\\)，事实性头的有效标签空间为 \\(\\mathcal{{Y}}_f=\\{{supported,unsupported\\}}\\)。给定基座评判器输出、文本特征、偏差风险和证据特征，BEA-Judge 学习任务特定概率分布：

        \\[
        p_\\theta(y\\mid x_i,h)=\\operatorname{{softmax}}(\\mathbf{{W}}_h\\mathbf{{z}}_i+\\mathbf{{b}}_h)_y,\\quad y\\in\\mathcal{{Y}}_h.
        \\]

        其中 \\(h\\in\\{{p,f\\}}\\) 表示任务头，\\(\\mathbf{{z}}_i\\) 是标准化后的融合特征向量。特征标准化仅使用训练集统计量：

        \\[
        z_{{ij}} = \\frac{{\\phi_j(x_i)-\\mu_j}}{{\\sigma_j+\\epsilon}}.
        \\]

        ### 4.2 基础 Judge 评分模块

        基础模块使用 M-Prometheus-3B 对成对样本产生真实评判输出，包含回答 A、B 的分数、分差、绝对 margin、基座预测标签及顺序置换诊断。对基座分差 \\(\\Delta s=s_A-s_B\\)，基座 margin 定义为：

        \\[
        m_i=|\\Delta s_i|.
        \\]

        该 margin 与基座预测标签共同表达基座评判器的相对确定性。

        ### 4.3 偏差感知模块

        偏差模块检测位置、长度、格式、rubric sensitivity 和数据源风险。每个风险项被裁剪到 \\([0,1]\\)，总体偏差风险定义为最大风险：

        \\[
        r_i^b=\\max\\{{r_i^{{pos}},r_i^{{len}},r_i^{{fmt}},r_i^{{rub}},r_i^{{src}}\\}}.
        \\]

        该模块不被解释为必然提升总体准确率的分类器组件，而是作为风险控制和复核优先级信号。

        ### 4.4 证据增强事实性模块

        事实性模块计算回答与上下文、参考答案之间的证据支持。令 \\(a\\) 为回答，\\(c\\) 为上下文，\\(r\\) 为参考答案，则回答支持度定义为：

        \\[
        S(a,c,r)=\\max(S(a,c\\oplus r),0.65S(a,c)+0.35S(a,r)).
        \\]

        模块进一步抽取数值缺失、日期缺失、实体缺失、实体别名缺失、否定错配、比较关系错配、低支持句比例和局部幻觉风险等特征。证据风险定义为相关局部风险的最大值：

        \\[
        r_i^e=\\max_k g_k(x_i).
        \\]

        ### 4.5 融合校准与置信度输出模块

        任务头使用 softmax 分类器，并通过开发集选择超参数。训练目标为带 \\(L_2\\) 正则的负对数似然：

        \\[
        \\mathcal{{L}}_h=-\\frac{{1}}{{N_h}}\\sum_i \\log p_\\theta(y_i\\mid x_i,h)+\\frac{{\\lambda}}{{2}}\\|\\mathbf{{W}}_h\\|_2^2.
        \\]

        模型选择目标为：

        \\[
        J=\\operatorname{{MacroF1}}+0.25\\operatorname{{Accuracy}}-0.05\\operatorname{{ECE}}.
        \\]

        温度缩放后概率为：

        \\[
        \\hat p_T(y\\mid x)=\\operatorname{{softmax}}\\left(\\frac{{\\log p_\\theta(y\\mid x)}}{{T}}\\right).
        \\]

        置信度定义为最大类别概率，风险分数定义为 \\(1-\\max_y \\hat p_T(y\\mid x)\\)，并结合偏差或证据风险触发 review_flag。

        ## 5 实验设置

        实验分为冻结四模块 BEA-Judge 与 QLoRA-BEA-Judge 两条线。冻结线使用 M-Prometheus-3B 真实输出和轻量融合校准头；QLoRA 线在 24GB RTX 4090 上使用 1024 token 长度、LoRA rank 16、三 seed（13、42、2026）训练。主指标包括 accuracy、macro-F1、ECE、Brier、Tie recall 和 review rate。所有温度、阈值、Tie policy 和 rescue policy 均在 dev 上选择，test 只用于最终报告。

        ## 6 实验结果与分析

        ### 6.1 冻结四模块主结果

        表2显示，冻结 BEA-Judge 在成对偏好测试集上达到 0.7512 accuracy、0.6730 macro-F1、0.0558 ECE 和 0.5231 Tie recall；事实性测试集上达到 0.7649 accuracy、0.7405 macro-F1 和 0.0377 ECE。

        {main_result_table()}

        ### 6.2 QLoRA 三 seed 结果

        表3显示，QLoRA-BEA-Judge 在 epoch1 accuracy-oriented 操作点上显著提升 accuracy、macro-F1 和 ECE，但 Tie recall 低于冻结基线。dev-selected tie-sensitive 操作点可以大幅提升 Tie recall，但牺牲 accuracy。epoch2 进一步提升 accuracy 和 macro-F1；加入 accuracy-constrained Tie rescue 后，在不降低 accuracy 的情况下提升 macro-F1 和 Tie recall。

        {qlora_table()}

        ![图2 主要操作点测试集指标对比](figures/fig2_main_operating_points.svg)

        ### 6.3 四模块消融

        表4与图3说明，基础 Judge 分数是成对偏好任务的核心信号，去除基础分数后 accuracy 降至 0.5626。事实性任务高度依赖证据增强；在 QLoRA 四模块回放中，去除证据模块后事实性 macro-F1 从 0.7405 降至 0.2629，ECE 升至 0.6274。Tie policy 对边界样本具有明显影响：去除 Tie policy 后总体 accuracy 上升至 0.8288，但 Tie recall 降至 0.3359，说明准确率与 Tie 召回之间存在明确权衡。

        {ablation_table()}

        ![图3 四模块消融结果](figures/fig3_four_module_ablation.svg)

        ### 6.4 训练轮数、SFT规模与外部基线

        训练消融显示，epoch2_1024 在 accuracy 与 macro-F1 上优于 epoch0.5 和 epoch1。SFT size 消融显示，25% 与 50% 数据已经能带来稳定增益，但 100% 数据仍取得最优宏平均表现。外部轻量基线比较中，QLoRA-BEA-Judge epoch2 相比 GRM-Llama3.2-3B、Qwen2.5-3B-Instruct 和 GLIDER 在 accuracy 与 macro-F1 上均有优势；其 ECE 也显著低于 GRM 与 Qwen。

        {external_table()}

        ![图4 QLoRA训练消融](figures/fig4_training_ablations.svg)

        ![图6 外部轻量基线比较](figures/fig6_external_baselines.svg)

        ### 6.5 风险复核分析

        风险覆盖曲线显示，成对头在复核约 49.95% 样本时捕获 85.11% 错误，自动接受部分 accuracy 达到 0.9260；事实性头在复核约 49.90% 样本时捕获 75.44% 错误，自动接受部分 accuracy 达到 0.8848。该结果表明，BEA-Judge 的风险分数可用于实际评审流程中的样本优先级排序。

        ![图5 风险阈值复核曲线](figures/fig5_risk_coverage.svg)

        ## 7 讨论

        事实性错误通常不是全局语义相似度不足，而是局部证据与回答之间存在实体、数值、日期、否定或比较关系错配。证据模块将这些局部错配显式纳入特征空间，使 softmax 头能够学习“回答整体相似但关键事实不被支持”的模式。相比之下，偏差模块的直接分类贡献并不稳定；它更适合用于识别 position、format 和 rubric sensitivity 等高风险子组，并把它们送入人工复核队列。

        Tie 策略体现了评判系统的操作点选择问题。若目标是最大化总体 accuracy，模型倾向减少 Tie；若目标是边界样本可靠性和人工可复核性，则 Tie-sensitive 或 Tie rescue 策略更合适。因此，论文和系统部署均应区分 accuracy-oriented 与 tie-sensitive 两种操作点，避免声称同一个操作点同时最优。

        ## 8 结论

        本文提出 BEA-Judge，一个基于真实 M-Prometheus-3B 输出的偏差感知证据增强评判校准框架。该框架通过基础 Judge 评分、偏差感知、证据增强事实性和融合校准四个模块，在 BEA-Judge-10K v2 上实现了可复核的生成式内容评估。冻结测试和三 seed QLoRA 实验表明，框架在 accuracy、macro-F1 和 ECE 上具有稳定优势；证据增强是事实性可靠性的主要来源，校准和 Tie 策略是成对偏好稳定性的关键。未来工作将扩大顺序置换评估、引入原子声明级事实标注，并在更多中文专业领域数据上验证风险复核策略。

        ## 数据与代码可用性

        处理后的 BEA-Judge-10K v2、源数据 manifest、预处理脚本、校验报告、模型输出表和图表应在投稿前归档至可签发 DOI 的仓库。HelpSteer2、OASST1、OffsetBias 和 RAGTruth 的再分发许可已通过项目审计；RewardBench 因混合子集许可限制未纳入正式训练。

        ## 参考文献

        {references()}
        """
    )


def make_tex() -> str:
    return clean_block(
        r"""
        \documentclass[UTF8,a4paper,zihao=-4]{ctexart}
        \usepackage{geometry}
        \usepackage{graphicx}
        \usepackage{booktabs}
        \usepackage{amsmath}
        \usepackage{hyperref}
        \usepackage{longtable}
        \geometry{left=2.6cm,right=2.6cm,top=2.5cm,bottom=2.5cm}
        \title{BEA-Judge：面向生成式人工智能内容评估的偏差感知证据增强评判校准框架}
        \author{作者信息待补充}
        \date{2026年5月30日}
        \begin{document}
        \maketitle

        \begin{abstract}
        生成式人工智能内容评估正在从静态基准测试转向可解释、可校准且可复核的评判系统。本文提出 BEA-Judge，一个建立在真实 M-Prometheus-3B 输出之上的偏差感知证据增强评判校准框架。该框架由基础 Judge 评分、偏差感知、证据增强事实性、融合校准与置信度输出四个模块组成。基于 BEA-Judge-10K v2 的冻结测试结果显示，成对偏好头达到 0.7512 accuracy、0.6730 macro-F1 和 0.0558 ECE；事实性头达到 0.7649 accuracy、0.7405 macro-F1 和 0.0377 ECE。三 seed QLoRA 实验进一步表明，epoch2 QLoRA-BEA-Judge 达到 0.8297 accuracy、0.7348 macro-F1 和 0.0278 ECE。消融实验表明，证据增强是事实性可靠性的主要贡献，校准与 Tie 策略是成对偏好稳定性的关键。
        \end{abstract}

        \noindent\textbf{关键词：} 生成式人工智能评估；LLM-as-a-Judge；偏差感知；事实性验证；概率校准；风险复核；QLoRA

        \section{引言}
        大语言模型输出的开放性使传统基于精确答案匹配的评估范式难以覆盖真实应用中的质量、偏好和事实性要求。LLM-as-a-Judge 通过将强模型或专用评判模型作为评审器，为开放式对话和响应排序提供了可扩展方案。本文关注成对偏好评估与事实性评估两类任务。前者需要处理偏好边界、Tie 标签和候选顺序敏感性，后者需要识别实体、数值、日期、否定和比较关系等局部证据错配。

        本文提出 BEA-Judge。与直接训练大型评判器不同，BEA-Judge 将 M-Prometheus-3B 的真实评判输出作为基座信号，再加入文本结构、偏差风险、证据支持和数据源特征，训练轻量级任务特定 softmax 头。随后，系统在开发集上选择温度缩放、数据集级温度策略、Tie 决策策略和低置信复核阈值。该设计将自动评判拆分为基础判断、风险诊断、概率校准和复核控制四个可审计环节。

        \section{相关工作}
        现有 LLM-as-a-Judge 研究证明强模型可近似人类偏好评估，但也暴露出位置偏差、长度偏差、格式偏差和校准不足等问题。Prometheus 与 M-Prometheus 提供了开放评判模型基础，RAGTruth 等事实性语料则揭示检索增强生成中的幻觉和证据不一致。本文不试图替代这些评判器，而是在其输出之上建立可诊断、可校准的轻量融合层。

        \section{数据集构建与分析}
        BEA-Judge-10K v2 由 10,200 条样本组成，包含开放问答 4,000 条、成对偏差评估 2,700 条、RAG 事实性评估 3,500 条。训练、开发和测试划分分别为 7,084、1,578 和 1,538 条。数据构建执行样本数、字段类型、枚举值、重复 ID、重复内容和跨 split 泄漏检查，正式结果门禁要求成对基座评判覆盖率为 6,946/6,946，启发式 fallback 行数为 0，未解析失败为 0。

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.98\linewidth]{figures/fig1_pipeline.svg}
        \caption{BEA-Judge 四模块框架与信息流。}
        \end{figure}

        \section{方法}
        对样本 $x_i$，成对偏好头的标签空间为 $\mathcal{Y}_p=\{A>B,B>A,Tie\}$，事实性头的有效标签空间为 $\mathcal{Y}_f=\{supported,unsupported\}$。给定基座评判器输出、文本特征、偏差风险和证据特征，BEA-Judge 学习任务特定概率分布：
        \[
        p_\theta(y\mid x_i,h)=\operatorname{softmax}(\mathbf{W}_h\mathbf{z}_i+\mathbf{b}_h)_y,\quad y\in\mathcal{Y}_h.
        \]
        偏差模块检测位置、长度、格式、rubric sensitivity 和数据源风险。证据模块抽取上下文/参考支持度、数值缺失、日期缺失、实体缺失、否定错配、比较关系错配和低支持句比例。融合校准模块使用开发集选择超参数、温度缩放、Tie policy 和复核阈值。

        \section{实验与分析}
        冻结 BEA-Judge 在成对偏好测试集上达到 0.7512 accuracy、0.6730 macro-F1、0.0558 ECE 和 0.5231 Tie recall；事实性测试集上达到 0.7649 accuracy、0.7405 macro-F1 和 0.0377 ECE。三 seed QLoRA 结果显示，epoch2 QLoRA-BEA-Judge 达到 0.8297 accuracy、0.7348 macro-F1 和 0.0278 ECE；加入 Tie rescue 后 macro-F1 提升至 0.7441，Tie recall 提升至 0.4795。

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.98\linewidth]{figures/fig2_main_operating_points.svg}
        \caption{主要操作点测试集指标对比。}
        \end{figure}

        消融实验表明，基础 Judge 分数是成对偏好任务的核心信号，去除基础分数后 accuracy 降至 0.5626。事实性任务高度依赖证据增强；在 QLoRA 四模块回放中，去除证据模块后事实性 macro-F1 从 0.7405 降至 0.2629。Tie policy 对边界样本具有明显影响，去除 Tie policy 后总体 accuracy 上升至 0.8288，但 Tie recall 降至 0.3359。

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.98\linewidth]{figures/fig3_four_module_ablation.svg}
        \caption{四模块消融结果。}
        \end{figure}

        风险覆盖曲线显示，成对头在复核约 49.95\% 样本时捕获 85.11\% 错误，事实性头在复核约 49.90\% 样本时捕获 75.44\% 错误。该结果表明，BEA-Judge 的风险分数可用于实际评审流程中的样本优先级排序。

        \section{结论}
        本文提出 BEA-Judge，一个基于真实 M-Prometheus-3B 输出的偏差感知证据增强评判校准框架。冻结测试和三 seed QLoRA 实验表明，该框架在 accuracy、macro-F1 和 ECE 上具有稳定优势；证据增强是事实性可靠性的主要来源，校准和 Tie 策略是成对偏好稳定性的关键。未来工作将扩大顺序置换评估、引入原子声明级事实标注，并在更多中文专业领域数据上验证风险复核策略。

        \section*{参考文献}
        参考文献完整列表见同目录 references.md，正式投稿前应转换为目标期刊格式。
        \end{document}
        """
    )


def write_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VSDX_DIR.mkdir(parents=True, exist_ok=True)
    fig1_pipeline()
    fig2_main_results()
    fig3_ablation()
    fig4_epoch_sft()
    fig5_risk_coverage()
    fig6_external_baseline()
    (OUT / "manuscript.md").write_text(make_paper_md(), encoding="utf-8", newline="\n")
    (OUT / "main.tex").write_text(make_tex(), encoding="utf-8", newline="\n")
    (OUT / "references.md").write_text(references() + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "title": "BEA-Judge：面向生成式人工智能内容评估的偏差感知证据增强评判校准框架",
        "generated_at": "2026-05-30",
        "reference_framework": str(ROOT / "论文参考" / "CodeUltraFeedback.pdf"),
        "outputs": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()),
        "note": "SVG files are editable vector graphics. VSDX files are best-effort minimal Visio packages generated with standard-library XML/ZIP only; verify in Microsoft Visio before journal submission.",
    }
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        textwrap.dedent(
            """
            # BEA-Judge中文论文_20260530

            本目录包含基于当前实验产物撰写的中文学术论文草稿与实验图表。

            - `manuscript.md`: 中文论文主稿。
            - `main.tex`: LaTeX骨架，正式投稿前可将Markdown正文迁移到目标模板。
            - `references.md`: 参考文献列表。
            - `figures/*.svg`: 可编辑矢量图，推荐作为Visio/Illustrator/Inkscape后续编辑源。
            - `vsdx/*.vsdx`: 标准库生成的最小Visio包，需在Microsoft Visio中打开核验。

            当前环境缺少 Microsoft Visio、LibreOffice、pandoc 和可验证的 `vsdx` Python库，因此未自动导出 DOCX/PDF，也无法本机验证 `.vsdx` 兼容性。
            """
        ).strip()
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    write_outputs()
    print(OUT)


if __name__ == "__main__":
    main()
