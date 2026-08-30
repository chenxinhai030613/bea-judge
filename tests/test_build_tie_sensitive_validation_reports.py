import argparse
import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_tie_sensitive_validation_reports import build_seed_report, metrics_for_rows  # noqa: E402


def pairwise_row(sample_id, dataset, gold, pred, p_a, p_b, p_tie):
    return {
        "id": sample_id,
        "dataset": dataset,
        "human_label": gold,
        "predicted_label": pred,
        "pairwise_label": pred,
        "final_score": 1.0 if pred == "A>B" else 0.5 if pred == "Tie" else 0.0,
        "confidence": max(p_a, p_b, p_tie),
        "risk_score": 1.0 - max(p_a, p_b, p_tie),
        "label_probabilities": {"A>B": p_a, "B>A": p_b, "Tie": p_tie},
    }


class BuildTieSensitiveValidationReportsTest(unittest.TestCase):
    def test_metrics_for_rows_reports_adjusted_tie_recall(self) -> None:
        rows = [
            pairwise_row("h1", "helpsteer2", "Tie", "Tie", 0.20, 0.10, 0.70),
            pairwise_row("h2", "helpsteer2", "Tie", "A>B", 0.70, 0.10, 0.20),
            pairwise_row("h3", "helpsteer2", "A>B", "A>B", 0.90, 0.05, 0.05),
        ]

        metrics = metrics_for_rows(rows)

        self.assertEqual(metrics["accuracy"], 0.6667)
        self.assertEqual(metrics["tie_recall"], 0.5)
        self.assertEqual(metrics["pred_distribution"]["Tie"], 1)

    def test_build_seed_report_writes_adjusted_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                pairwise_row("h1", "helpsteer2", "Tie", "A>B", 0.60, 0.05, 0.35),
                pairwise_row("h2", "helpsteer2", "Tie", "A>B", 0.62, 0.05, 0.33),
                pairwise_row("h3", "helpsteer2", "A>B", "A>B", 0.85, 0.03, 0.12),
                pairwise_row("h4", "helpsteer2", "B>A", "B>A", 0.03, 0.85, 0.12),
                pairwise_row("m1", "mt_bench", "A>B", "A>B", 0.90, 0.03, 0.07),
                pairwise_row("m2", "mt_bench", "B>A", "B>A", 0.03, 0.90, 0.07),
            ]
            validation_report = {
                "heads": {"pairwise": {"calibrated_dev_metrics": {}}},
                "test_evaluation": {"pairwise": {"metrics": {}}},
                "train_evaluation": {"pairwise": {"metrics": {}}},
            }
            calibrated_results = {
                "train": {"pairwise": rows},
                "dev": {"pairwise": rows},
                "test": {"pairwise": rows},
            }
            validation_path = tmp_path / "validation_report.json"
            calibrated_path = tmp_path / "calibrated_results.json"
            validation_path.write_text(json.dumps(validation_report), encoding="utf-8")
            calibrated_path.write_text(json.dumps(calibrated_results), encoding="utf-8")
            args = argparse.Namespace(
                dataset="helpsteer2",
                macro_gain_min=0.01,
                ece_max=0.50,
                thresholds=[0.30, 0.40],
                margins=[1.0],
                output_template=str(tmp_path / "out_seed{seed}"),
            )

            run_dir, policy = build_seed_report(
                seed="unit",
                validation_report=validation_path,
                calibrated_results=calibrated_path,
                frozen_dev={"accuracy": 0.50, "macro_f1": 0.50},
                args=args,
            )

            adjusted_report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
            adjusted_rows = json.loads((run_dir / "calibrated_results.json").read_text(encoding="utf-8"))
            self.assertEqual(policy["policy"]["min_tie_probability"], 0.30)
            self.assertEqual(adjusted_report["test_evaluation"]["pairwise"]["metrics"]["tie_recall"], 1.0)
            self.assertEqual(adjusted_rows["test"]["pairwise"][0]["predicted_label"], "Tie")


if __name__ == "__main__":
    unittest.main()
