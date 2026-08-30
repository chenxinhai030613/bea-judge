import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_qlora_submission_package import validate_submission_summary  # noqa: E402


def metric(mean, std=0.0):
    return {"n": 3, "mean": mean, "std": std, "min": mean, "max": mean}


def valid_summary():
    return {
        "baseline": {
            "metrics": {
                "accuracy": metric(0.7512),
                "macro_f1": metric(0.6730),
                "ece": metric(0.0558),
                "tie_recall": metric(0.5231),
            }
        },
        "operating_points": {
            "accuracy_oriented": {
                "gate": {"all_passed": False},
                "metrics": {
                    "accuracy": metric(0.8025),
                    "macro_f1": metric(0.7128),
                    "ece": metric(0.0279),
                    "tie_recall": metric(0.4538),
                },
            },
            "tie_sensitive_dev_selected": {
                "gate": {"all_passed": True},
                "metrics": {
                    "accuracy": metric(0.7582),
                    "macro_f1": metric(0.7169),
                    "ece": metric(0.0229),
                    "tie_recall": metric(0.7667),
                },
            },
        },
    }


class ValidateQloraSubmissionPackageTest(unittest.TestCase):
    def test_validate_submission_summary_accepts_dual_operating_point_package(self) -> None:
        result = validate_submission_summary(valid_summary())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed"], [])
        self.assertTrue(result["checks"]["tie_point_gate_all_passed"])
        self.assertTrue(result["checks"]["tie_point_tie_recall_above_baseline"])

    def test_validate_submission_summary_rejects_missing_tie_improvement(self) -> None:
        summary = valid_summary()
        summary["operating_points"]["tie_sensitive_dev_selected"]["metrics"]["tie_recall"] = metric(0.50)

        result = validate_submission_summary(summary)

        self.assertFalse(result["passed"])
        self.assertIn("tie_point_tie_recall_above_baseline", result["failed"])

    def test_validate_submission_summary_requires_both_operating_points(self) -> None:
        summary = valid_summary()
        del summary["operating_points"]["tie_sensitive_dev_selected"]

        with self.assertRaisesRegex(ValueError, "both operating points"):
            validate_submission_summary(summary)


if __name__ == "__main__":
    unittest.main()
