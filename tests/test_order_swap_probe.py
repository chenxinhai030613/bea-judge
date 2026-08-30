import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from order_swap_probe import (  # noqa: E402
    build_consistency_rows,
    hard_example_rows,
    invert_pairwise_label,
    run_probe,
    select_probe_entries,
    summarize_consistency,
)


def sample(sample_id, dataset="mt_bench", split="dev", gold="A>B"):
    return {
        "id": sample_id,
        "dataset": dataset,
        "task_type": "open_qa",
        "prompt": "Which answer is better?",
        "context": None,
        "answer_a": "A answer",
        "answer_b": "B answer",
        "reference": None,
        "human_label": gold,
        "split": split,
        "metadata": {},
    }


def calibrated(sample_id, dataset="mt_bench", gold="A>B", pred="A>B", confidence=0.8):
    return {
        "id": sample_id,
        "dataset": dataset,
        "task_type": "open_qa",
        "split": "dev",
        "head": "pairwise",
        "human_label": gold,
        "predicted_label": pred,
        "confidence": confidence,
    }


def base_score(sample_id, pred="A>B", score_a=1.0, score_b=0.0):
    return {
        "id": sample_id,
        "dataset": "mt_bench",
        "task_type": "open_qa",
        "judge_backend": "m_prometheus",
        "parse_status": "ok",
        "pred_label": pred,
        "parsed_scores": {"score_a": score_a, "score_b": score_b},
    }


class OrderSwapProbeTest(unittest.TestCase):
    def test_invert_pairwise_label(self):
        self.assertEqual(invert_pairwise_label("A>B"), "B>A")
        self.assertEqual(invert_pairwise_label("B>A"), "A>B")
        self.assertEqual(invert_pairwise_label("Tie"), "Tie")
        self.assertIsNone(invert_pairwise_label("supported"))

    def test_select_probe_entries_filters_to_dev_hard_target_rows(self):
        samples = [
            sample("s1", "mt_bench", "dev", "A>B"),
            sample("s2", "pandalm", "dev", "A>B"),
            sample("s3", "other", "dev", "A>B"),
            sample("s4", "mt_bench", "test", "A>B"),
        ]
        calibrated_payload = {
            "dev": {
                "pairwise": [
                    calibrated("s1", "mt_bench", "A>B", "B>A", 0.55),
                    calibrated("s2", "pandalm", "A>B", "A>B", 0.95),
                    calibrated("s3", "other", "A>B", "B>A", 0.50),
                    calibrated("s4", "mt_bench", "A>B", "B>A", 0.50),
                ]
            }
        }
        base_scores = [base_score("s1"), base_score("s2"), base_score("s3"), base_score("s4")]

        entries = select_probe_entries(
            samples=samples,
            calibrated=calibrated_payload,
            base_scores=base_scores,
            target_datasets=["mt_bench", "pandalm"],
            low_confidence_threshold=0.70,
            per_dataset_limit=5,
        )

        self.assertEqual([entry["id"] for entry in entries], ["s1"])
        self.assertIn("calibrated_error", entries[0]["selection_reasons"])
        self.assertIn("low_confidence", entries[0]["selection_reasons"])

    def test_build_consistency_rows_and_summary(self):
        entries = [
            {
                "id": "s1",
                "sample": sample("s1", "mt_bench", "dev", "A>B"),
                "base": base_score("s1", "A>B", 1.0, 0.0),
                "calibrated": calibrated("s1", "mt_bench", "A>B", "B>A", 0.55),
                "selection_reasons": ["calibrated_error"],
            },
            {
                "id": "s2",
                "sample": sample("s2", "mt_bench", "dev", "B>A"),
                "base": base_score("s2", "A>B", 1.0, 0.0),
                "calibrated": calibrated("s2", "mt_bench", "B>A", "B>A", 0.80),
                "selection_reasons": ["base_calibrated_disagreement"],
            },
        ]
        swap_scores = [
            {"id": "s1", "pred_label": "B>A", "parsed_scores": {"score_a": 0.0, "score_b": 1.0}},
            {"id": "s2", "pred_label": "A>B", "parsed_scores": {"score_a": 1.0, "score_b": 0.0}},
        ]

        rows = build_consistency_rows(entries, swap_scores)
        summary = summarize_consistency(rows)
        overall = [row for row in summary if row["dataset"] == "overall"][0]

        self.assertEqual(rows[0]["swap_consistency_flag"], 1.0)
        self.assertEqual(rows[1]["swap_consistency_flag"], 0.0)
        self.assertEqual(overall["swap_consistency_rate"], 0.5)
        self.assertEqual(overall["error_rate_when_inconsistent"], 0.0)

    def test_hard_example_rows_include_swap_diagnostics(self):
        entries = [
            {
                "id": "s1",
                "sample": sample("s1", "mt_bench", "dev", "A>B"),
                "base": base_score("s1", "A>B", 1.0, 0.0),
                "calibrated": calibrated("s1", "mt_bench", "A>B", "B>A", 0.55),
                "selection_reasons": ["calibrated_error"],
            }
        ]
        consistency = [
            {
                "id": "s1",
                "swap_pred_label": "B>A",
                "swap_consistency_flag": 1.0,
                "swap_margin_delta": 0.0,
            }
        ]

        rows = hard_example_rows(entries, consistency)

        self.assertEqual(rows[0]["id"], "s1")
        self.assertEqual(rows[0]["swap_pred_label"], "B>A")
        self.assertEqual(rows[0]["selection_reasons"], "calibrated_error")

    def test_run_probe_dry_run_writes_selection_and_tables(self):
        samples = [sample("s1", "mt_bench", "dev", "A>B")]
        calibrated_payload = {
            "dev": {"pairwise": [calibrated("s1", "mt_bench", "A>B", "B>A", 0.55)]}
        }
        base_scores = [base_score("s1", "A>B", 1.0, 0.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "dataset.json"
            calibrated_path = tmp / "calibrated.json"
            base_path = tmp / "base_scores.json"
            output_dir = tmp / "probe"
            input_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            calibrated_path.write_text(json.dumps(calibrated_payload), encoding="utf-8")
            base_path.write_text(json.dumps(base_scores), encoding="utf-8")

            report = run_probe(
                input_path=input_path,
                calibrated_results_path=calibrated_path,
                base_scores_path=base_path,
                output_dir=output_dir,
                target_datasets=["mt_bench"],
                low_confidence_threshold=0.70,
                per_dataset_limit=5,
                dry_run=True,
                backend="m_prometheus",
                model_path="mock",
                device="auto",
                max_new_tokens=16,
            )

            self.assertEqual(report["selection"]["selected_count"], 1)
            self.assertTrue((output_dir / "selected_swap_samples.json").exists())
            self.assertTrue((output_dir / "swap_consistency_table.csv").exists())
            self.assertTrue((output_dir / "hard_examples_table.md").exists())


if __name__ == "__main__":
    unittest.main()
