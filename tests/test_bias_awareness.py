import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_awareness import (
    build_bias_profile,
    favor_side,
    length_bias_risk,
    summarize_bias_profiles,
)


class BiasAwarenessTest(unittest.TestCase):
    def test_length_bias_flags_prediction_that_favors_longer_wrong_answer(self) -> None:
        sample = {
            "id": "length-1",
            "dataset": "unit",
            "task_type": "open_qa",
            "answer_a": "Short but correct.",
            "answer_b": "Longer answer with many extra words that should not be preferred.",
            "human_label": "A>B",
            "metadata": {},
        }
        prediction = {"predicted_label": "B>A", "confidence": 0.8}

        risk, reasons = length_bias_risk(sample, prediction)

        self.assertEqual(risk, 1.0)
        self.assertIn("prediction_favors_longer_answer_against_gold", reasons)

    def test_position_perturbation_mismatch_triggers_review(self) -> None:
        sample = {
            "id": "position-1",
            "dataset": "synthetic_perturbed",
            "task_type": "pairwise_bias",
            "answer_a": "A better answer.",
            "answer_b": "A weaker answer.",
            "human_label": "A>B",
            "metadata": {"bias_type": "position", "perturbation_applied": "position"},
        }
        prediction = {"predicted_label": "B>A", "confidence": 0.7}

        profile = build_bias_profile(sample, prediction)

        self.assertEqual(profile["bias"]["position_risk"], 1.0)
        self.assertTrue(profile["bias"]["review_required"])
        self.assertIn("position_perturbation_prediction_mismatch", profile["bias"]["reasons"])

    def test_summary_reports_accuracy_by_dataset_and_bias_type(self) -> None:
        profiles = [
            {
                "id": "a",
                "dataset": "unit_a",
                "bias": {"overall_bias_risk": 0.2, "review_required": False},
                "prediction": {"predicted_label": "A>B", "gold_label": "A>B"},
                "metadata": {"bias_type": "length"},
            },
            {
                "id": "b",
                "dataset": "unit_a",
                "bias": {"overall_bias_risk": 0.8, "review_required": True},
                "prediction": {"predicted_label": "B>A", "gold_label": "A>B"},
                "metadata": {"bias_type": "length"},
            },
            {
                "id": "c",
                "dataset": "unit_b",
                "bias": {"overall_bias_risk": 0.0, "review_required": False},
                "prediction": {"predicted_label": "Tie", "gold_label": "Tie"},
                "metadata": {"bias_type": "none"},
            },
        ]

        summary = summarize_bias_profiles(profiles)

        self.assertEqual(summary["overall"]["profile_count"], 3)
        self.assertEqual(summary["by_dataset"]["unit_a"]["accuracy"], 0.5)
        self.assertEqual(summary["by_dataset"]["unit_b"]["accuracy"], 1.0)
        self.assertEqual(summary["by_bias_type"]["length"]["review_rate"], 0.5)

    def test_favor_side_handles_pairwise_and_tie_labels(self) -> None:
        self.assertEqual(favor_side("A>B"), "a")
        self.assertEqual(favor_side("B>A"), "b")
        self.assertEqual(favor_side("Tie"), "tie")
        self.assertIsNone(favor_side("supported"))


if __name__ == "__main__":
    unittest.main()
