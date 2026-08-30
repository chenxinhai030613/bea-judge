import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_qlora_experiments import build_comparison, raw_pairwise_metrics  # noqa: E402


class CompareQloraExperimentsTest(unittest.TestCase):
    def test_raw_pairwise_metrics_include_tie_recall(self) -> None:
        rows = [
            {"gold_label": "A>B", "pred_label": "A>B", "parsed_scores": {"score_a": 1.0, "score_b": 0.0}},
            {"gold_label": "B>A", "pred_label": "B>A", "parsed_scores": {"score_a": 0.0, "score_b": 1.0}},
            {"gold_label": "Tie", "pred_label": "Tie", "parsed_scores": {"score_a": 0.5, "score_b": 0.5}},
            {"gold_label": "Tie", "pred_label": "A>B", "parsed_scores": {"score_a": 1.0, "score_b": 0.0}},
        ]

        metrics = raw_pairwise_metrics(rows)

        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["tie_recall"], 0.5)
        self.assertEqual(metrics["parse_failure_rate"], 0.0)

    def test_build_comparison_gate_uses_requested_thresholds(self) -> None:
        frozen_report = {
            "test_evaluation": {
                "pairwise": {
                    "metrics": {
                        "accuracy": 0.7512,
                        "macro_f1": 0.673,
                        "ece": 0.0558,
                        "tie_recall": 0.5231,
                    }
                }
            }
        }
        qlora_report = {
            "test_evaluation": {
                "pairwise": {
                    "metrics": {
                        "accuracy": 0.79,
                        "macro_f1": 0.70,
                        "ece": 0.05,
                        "tie_recall": 0.60,
                    }
                }
            }
        }
        raw_rows = [
            {"gold_label": "A>B", "pred_label": "A>B", "parsed_scores": {"score_a": 1.0, "score_b": 0.0}},
            {"gold_label": "B>A", "pred_label": "B>A", "parsed_scores": {"score_a": 0.0, "score_b": 1.0}},
            {"gold_label": "Tie", "pred_label": "Tie", "parsed_scores": {"score_a": 0.5, "score_b": 0.5}},
        ]

        result = build_comparison(
            frozen_report=frozen_report,
            qlora_report=qlora_report,
            raw_frozen_summary=None,
            raw_qlora_rows=raw_rows,
        )

        self.assertTrue(result["gate"]["checks"]["qlora_raw_macro_f1_gain_min_0_10"])
        self.assertTrue(result["gate"]["checks"]["qlora_bea_macro_f1_gain_min_0_02"])
        self.assertTrue(result["gate"]["checks"]["tie_recall_not_below_baseline"])
        self.assertTrue(result["gate"]["checks"]["ece_max_0_06"])


if __name__ == "__main__":
    unittest.main()
