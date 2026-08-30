import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_qlora_base_judge import repair_failed_rows, summarize  # noqa: E402


class RunQloraBaseJudgeTest(unittest.TestCase):
    def test_repair_failed_rows_uses_natural_language_parser(self) -> None:
        rows = [
            {
                "id": "tie-1",
                "pred_label": None,
                "pred_score": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
                "raw_output": "Both responses are identical in content and structure, providing a complete and accurate description.",
                "parse_metadata": {"raw": "Both responses are identical in content and structure, providing a complete and accurate description."},
            },
            {
                "id": "keep-1",
                "pred_label": "A>B",
                "pred_score": 1.0,
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
                "raw_output": "[RESULT] A",
                "parse_metadata": {"raw": "[RESULT] A", "label": "A>B"},
            },
        ]

        repaired = repair_failed_rows(rows)

        self.assertEqual(repaired[0]["pred_label"], "Tie")
        self.assertEqual(repaired[0]["pred_score"], 0.0)
        self.assertEqual(repaired[0]["parsed_scores"], {"score_a": 0.5, "score_b": 0.5})
        self.assertEqual(repaired[0]["parse_status"], "retry_reparse_ok")
        self.assertEqual(repaired[1]["pred_label"], "A>B")

    def test_summarize_counts_repaired_rows_as_valid(self) -> None:
        rows = [
            {
                "pred_label": "Tie",
                "gold_label": "Tie",
            },
            {
                "pred_label": "A>B",
                "gold_label": "B>A",
            },
        ]

        summary = summarize(rows, {"available": True})

        self.assertEqual(summary["parse_failure_count"], 0)
        self.assertEqual(summary["coverage"]["parsed_rows"], 2)
        self.assertEqual(summary["overall"]["pairwise_accuracy"], 0.5)

    def test_cli_exposes_retry_token_budget(self) -> None:
        content = (ROOT / "scripts" / "run_qlora_base_judge.py").read_text(encoding="utf-8")

        self.assertIn("--retry-max-new-tokens", content)
        self.assertIn("retry_max_new_tokens", content)


if __name__ == "__main__":
    unittest.main()
