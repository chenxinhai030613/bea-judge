import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tie_sensitive_policy_audit import apply_policy_to_rows, select_policy  # noqa: E402


def pairwise_row(sample_id, dataset, gold, pred, p_a, p_b, p_tie):
    return {
        "id": sample_id,
        "dataset": dataset,
        "human_label": gold,
        "predicted_label": pred,
        "label_probabilities": {"A>B": p_a, "B>A": p_b, "Tie": p_tie},
    }


class TieSensitivePolicyAuditTest(unittest.TestCase):
    def test_apply_policy_only_changes_target_dataset_tie_candidates(self) -> None:
        rows = [
            pairwise_row("h1", "helpsteer2", "Tie", "A>B", 0.60, 0.05, 0.35),
            pairwise_row("h2", "helpsteer2", "A>B", "A>B", 0.80, 0.05, 0.15),
            pairwise_row("m1", "mt_bench", "Tie", "A>B", 0.60, 0.05, 0.35),
        ]

        metrics = apply_policy_to_rows(
            rows,
            dataset="helpsteer2",
            min_tie_probability=0.30,
            max_ab_margin=1.0,
        )

        self.assertEqual(metrics["accuracy"], 0.666667)
        self.assertEqual(metrics["tie_recall"], 0.5)
        self.assertEqual(metrics["tie_pred_count"], 1)

    def test_select_policy_uses_dev_quality_constraints(self) -> None:
        rows = [
            pairwise_row("h1", "helpsteer2", "Tie", "A>B", 0.60, 0.05, 0.35),
            pairwise_row("h2", "helpsteer2", "Tie", "A>B", 0.62, 0.05, 0.33),
            pairwise_row("h3", "helpsteer2", "A>B", "A>B", 0.85, 0.03, 0.12),
            pairwise_row("h4", "helpsteer2", "B>A", "B>A", 0.03, 0.85, 0.12),
            pairwise_row("m1", "mt_bench", "A>B", "A>B", 0.90, 0.03, 0.07),
            pairwise_row("m2", "mt_bench", "B>A", "B>A", 0.03, 0.90, 0.07),
        ]

        policy, candidates = select_policy(
            rows,
            dataset="helpsteer2",
            frozen_dev={"accuracy": 0.50, "macro_f1": 0.50},
            macro_gain_min=0.01,
            ece_max=0.50,
            thresholds=[0.30, 0.40],
            margins=[1.0],
        )

        self.assertIsNotNone(policy)
        self.assertEqual(policy["min_tie_probability"], 0.30)
        self.assertEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
