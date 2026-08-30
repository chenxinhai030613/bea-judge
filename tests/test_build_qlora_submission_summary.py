import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_qlora_submission_summary import build_submission_summary  # noqa: E402


def metric(mean, std=0.0):
    return {"n": 3, "mean": mean, "std": std, "min": mean, "max": mean}


class BuildQloraSubmissionSummaryTest(unittest.TestCase):
    def test_build_submission_summary_separates_operating_points(self) -> None:
        conservative = {
            "gate": {"passed_count": 0, "total": 3},
            "by_system": {
                "Current BEA-Judge": {
                    "accuracy": metric(0.75),
                    "macro_f1": metric(0.67),
                    "ece": metric(0.05),
                    "tie_recall": metric(0.52),
                },
                "QLoRA-BEA-Judge": {
                    "accuracy": metric(0.80),
                    "macro_f1": metric(0.71),
                    "ece": metric(0.03),
                    "tie_recall": metric(0.45),
                },
            },
        }
        tie_sensitive = {
            "gate": {"passed_count": 3, "total": 3},
            "by_system": {
                "QLoRA-BEA-Judge": {
                    "accuracy": metric(0.76),
                    "macro_f1": metric(0.72),
                    "ece": metric(0.02),
                    "tie_recall": metric(0.77),
                },
            },
        }

        summary = build_submission_summary(
            conservative_summary=conservative,
            tie_sensitive_summary=tie_sensitive,
        )

        self.assertEqual(summary["operating_points"]["accuracy_oriented"]["gate"]["passed_count"], 0)
        self.assertEqual(summary["operating_points"]["tie_sensitive_dev_selected"]["gate"]["passed_count"], 3)
        self.assertEqual(
            summary["operating_points"]["accuracy_oriented"]["delta_vs_current"]["macro_f1"],
            0.04,
        )
        self.assertEqual(
            summary["operating_points"]["tie_sensitive_dev_selected"]["delta_vs_current"]["tie_recall"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
