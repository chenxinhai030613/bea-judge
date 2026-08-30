from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = next(ROOT.glob("QLoRA-BEA-Judge_SCI论文_20260531"))
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_SVG = FIG_DIR / "fig1_bea_judge_framework_ieee_final.svg"
OUT_PNG = FIG_DIR / "fig1_bea_judge_framework_ieee_final.png"


def text(lines: list[str], x: float, y: float, items: list[str], size: float = 12, bold_first: bool = False) -> None:
    for i, item in enumerate(items):
        weight = "700" if bold_first and i == 0 else "400"
        lines.append(
            f'  <text x="{x:.1f}" y="{y + i * (size + 3):.1f}" '
            f'class="txt" font-size="{size:.1f}" font-weight="{weight}" '
            f'text-anchor="middle">{escape(item)}</text>'
        )


def main() -> None:
    nodes = [
        ("raw", 70, 112, 150, 58, ["(1) Pairwise samples", "prompt, response A/B"], "#fff"),
        ("gates", 250, 112, 150, 58, ["(2) Quality gates", "license, bias, factuality"], "#f2f2f2"),
        ("split", 430, 112, 150, 58, ["(3) Stratified split", "train / dev / test"], "#fff"),
        ("backbone", 650, 112, 150, 58, ["(4) 3B judge", "backbone"], "#fff"),
        ("qlora", 830, 112, 150, 58, ["(5) QLoRA SFT", "pairwise objective"], "#e2e2e2"),
        ("ckpt", 1010, 112, 150, 58, ["(6) Seeds and", "checkpoints"], "#fff"),
        ("dev", 1230, 112, 150, 58, ["(7) Dev metrics", "accuracy, tie recall"], "#fff"),
        ("calib", 1410, 112, 170, 58, ["(8) Calibration", "temperature, thresholds"], "#f2f2f2"),
        ("score", 110, 308, 170, 62, ["Base judge scoring", "score_A, score_B, margin"], "#fff"),
        ("bias", 345, 308, 170, 62, ["Bias-aware features", "position, length, format"], "#f2f2f2"),
        ("fact", 580, 308, 170, 62, ["Evidence factuality", "entity, number, date gaps"], "#fff"),
        ("fusion", 815, 308, 170, 62, ["Fusion head", "probability and confidence"], "#e2e2e2"),
        ("tie", 1050, 308, 170, 62, ["Tie rescue", "dev-only policy search"], "#f2f2f2"),
        ("output", 1285, 308, 220, 62, ["Structured output", "label, risk score, review flag"], "#fff"),
        ("lock", 130, 528, 185, 62, ["Locked test split", "used once"], "#fff"),
        ("internal", 405, 528, 185, 62, ["Internal runs", "3-seed mean +/- std"], "#f2f2f2"),
        ("external", 680, 528, 185, 62, ["External baselines", "single full-test run"], "#f2f2f2"),
        ("metrics", 955, 528, 185, 62, ["Report metrics", "accuracy, ECE, risk"], "#fff"),
        ("claim", 1230, 528, 170, 62, ["Final tables", "and figures"], "#fff"),
    ]
    idx = {node[0]: node for node in nodes}

    def point(node_id: str, side: str) -> tuple[float, float]:
        _, x, y, w, h, _, _ = idx[node_id]
        if side == "L":
            return x, y + h / 2
        if side == "R":
            return x + w, y + h / 2
        if side == "T":
            return x + w / 2, y
        if side == "B":
            return x + w / 2, y + h
        raise ValueError(side)

    def arrow(lines: list[str], start: str, start_side: str, end: str, end_side: str, soft: bool = False) -> None:
        x1, y1 = point(start, start_side)
        x2, y2 = point(end, end_side)
        cls = "edge-soft" if soft else "edge"
        lines.append(f'  <path class="{cls}" d="M {x1:.1f},{y1:.1f} L {x2:.1f},{y2:.1f}"/>')

    def elbow(
        lines: list[str],
        start: str,
        start_side: str,
        end: str,
        end_side: str,
        mid_x: float,
        soft: bool = False,
    ) -> None:
        x1, y1 = point(start, start_side)
        x2, y2 = point(end, end_side)
        cls = "edge-soft" if soft else "edge"
        lines.append(f'  <path class="{cls}" d="M {x1:.1f},{y1:.1f} L {mid_x:.1f},{y1:.1f} L {mid_x:.1f},{y2:.1f} L {x2:.1f},{y2:.1f}"/>')

    def routed(
        lines: list[str],
        start: str,
        start_side: str,
        end: str,
        end_side: str,
        waypoints: list[tuple[float, float]],
        soft: bool = False,
    ) -> None:
        x1, y1 = point(start, start_side)
        x2, y2 = point(end, end_side)
        pts = [(x1, y1), *waypoints, (x2, y2)]
        cls = "edge-soft" if soft else "edge"
        d = " ".join(f"L {x:.1f},{y:.1f}" if i else f"M {x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        lines.append(f'  <path class="{cls}" d="{d}"/>')

    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="760" viewBox="0 0 1680 760">',
        "  <defs>",
        '    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#111"/></marker>',
        "    <style><![CDATA[",
        '      .txt { font-family: "Times New Roman", SimSun, serif; fill: #111; }',
        "      .box { stroke: #111; stroke-width: 1.1; rx: 6; }",
        "      .group { fill: none; stroke: #555; stroke-width: 0.9; stroke-dasharray: 6 4; rx: 8; }",
        "      .edge { fill: none; stroke: #111; stroke-width: 1.1; marker-end: url(#arrow); }",
        "      .edge-soft { fill: none; stroke: #111; stroke-width: 1.0; stroke-dasharray: 5 4; marker-end: url(#arrow); }",
        "    ]]></style>",
        "  </defs>",
        '  <rect width="1680" height="760" fill="#fff"/>',
        '  <text x="840" y="42" class="txt" font-size="18" font-weight="700" text-anchor="middle">Fig. 1. Experimental workflow of the QLoRA-BEA-Judge framework</text>',
    ]

    groups = [
        (45, 72, 560, 125, "A. Data construction"),
        (625, 72, 560, 125, "B. QLoRA training"),
        (1205, 72, 430, 125, "C. Development selection"),
        (45, 268, 1590, 142, "D. Inference and calibrated judgment"),
        (45, 488, 1590, 140, "E. Locked evaluation and reporting"),
    ]
    for x, y, w, h, label in groups:
        lines.append(f'  <rect class="group" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        lines.append(f'  <text x="{x + 12}" y="{y + 20}" class="txt" font-size="13" font-weight="700">{escape(label)}</text>')

    for _, x, y, w, h, label_lines, fill in nodes:
        lines.append(f'  <rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>')
        text(lines, x + w / 2, y + 24, label_lines, size=12, bold_first=True)

    for edge in [
        ("raw", "R", "gates", "L"),
        ("gates", "R", "split", "L"),
        ("split", "R", "backbone", "L"),
        ("backbone", "R", "qlora", "L"),
        ("qlora", "R", "ckpt", "L"),
        ("ckpt", "R", "dev", "L"),
        ("dev", "R", "calib", "L"),
        ("score", "R", "bias", "L"),
        ("bias", "R", "fact", "L"),
        ("fact", "R", "fusion", "L"),
        ("fusion", "R", "tie", "L"),
        ("tie", "R", "output", "L"),
        ("lock", "R", "internal", "L"),
        ("internal", "R", "external", "L"),
        ("external", "R", "metrics", "L"),
        ("metrics", "R", "claim", "L"),
    ]:
        arrow(lines, *edge)

    elbow(lines, "calib", "B", "fusion", "T", 1520, soft=True)
    routed(lines, "ckpt", "B", "score", "L", [(1085, 238), (95, 238), (95, 339)], soft=False)
    routed(lines, "split", "B", "lock", "L", [(505, 236), (32, 236), (32, 559)], soft=True)
    elbow(lines, "output", "B", "metrics", "T", 1395)

    lines.extend(
        [
            '  <line x1="90" y1="682" x2="170" y2="682" class="edge"/>',
            '  <text x="192" y="687" class="txt" font-size="12">model/data flow</text>',
            '  <line x1="440" y1="682" x2="520" y2="682" class="edge-soft"/>',
            '  <text x="542" y="687" class="txt" font-size="12">protocol constraints selected on dev</text>',
            '  <text x="840" y="724" class="txt" font-size="12" text-anchor="middle">Test discipline: thresholds and Tie rescue are fixed before the locked test report.</text>',
            "</svg>",
        ]
    )

    OUT_SVG.write_text("\n".join(lines), encoding="utf-8")

    import cairosvg

    cairosvg.svg2png(url=str(OUT_SVG), write_to=str(OUT_PNG), scale=2)
    print(f"SVG={OUT_SVG}")
    print(f"PNG={OUT_PNG}")


if __name__ == "__main__":
    main()
