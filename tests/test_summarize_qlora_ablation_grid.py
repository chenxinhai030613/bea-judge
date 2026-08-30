import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_qlora_ablation_grid import build_summary, markdown_summary  # noqa: E402


def report(raw_acc: float, qlora_acc: float) -> dict:
    return {
        "comparison_rows": [
            {"system": "Raw M-Prometheus-3B", "accuracy": 0.56, "macro_f1": 0.40, "ece": "", "tie_recall": ""},
            {"system": "Current BEA-Judge", "accuracy": 0.75, "macro_f1": 0.67, "ece": 0.05, "tie_recall": 0.52},
            {"system": "QLoRA-M-Prometheus-3B", "accuracy": raw_acc, "macro_f1": 0.68, "ece": 0.18, "tie_recall": 0.22},
            {"system": "QLoRA-BEA-Judge", "accuracy": qlora_acc, "macro_f1": 0.71, "ece": 0.03, "tie_recall": 0.45},
        ]
    }


class SummarizeQloraAblationGridTest(unittest.TestCase):
    def test_build_summary_aggregates_multiple_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for setting, seed, raw_acc, qlora_acc in [
                ("epoch0p5_1024", "13", 0.80, 0.78),
                ("epoch0p5_1024", "42", 0.81, 0.79),
                ("epoch1_1024", "13", 0.82, 0.80),
                ("epoch1_1024", "42", 0.83, 0.81),
            ]:
                path = tmp_path / f"{setting}_{seed}.json"
                path.write_text(json.dumps(report(raw_acc, qlora_acc)), encoding="utf-8")

            summary = build_summary(
                settings=["epoch0p5_1024", "epoch1_1024"],
                seeds=["13", "42"],
                template=str(tmp_path / "{setting}_{seed}.json"),
            )

            self.assertEqual(summary["settings"], ["epoch0p5_1024", "epoch1_1024"])
            self.assertEqual(
                summary["results"][0]["by_system"]["QLoRA-BEA-Judge"]["accuracy"]["mean"],
                0.785,
            )
            markdown = markdown_summary(summary, "Unit Title")
            self.assertIn("epoch0p5_1024", markdown)
            self.assertIn("QLoRA-BEA-Judge", markdown)


if __name__ == "__main__":
    unittest.main()
