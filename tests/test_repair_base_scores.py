import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from repair_base_scores import coverage_report, failed_ids, merge_repaired_rows, reparse_retry_rows


class RepairBaseScoresTest(unittest.TestCase):
    def test_failed_ids_returns_only_invalid_rows(self) -> None:
        rows = [
            {
                "id": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            },
            {
                "id": "failed",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
            },
        ]

        self.assertEqual(failed_ids(rows), ["failed"])

    def test_merge_repaired_rows_replaces_failed_id_only(self) -> None:
        required = ["ok", "failed"]
        original = [
            {
                "id": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            },
            {
                "id": "failed",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
            },
        ]
        retry = [
            {
                "id": "failed",
                "pred_label": "B>A",
                "parsed_scores": {"score_a": 0.0, "score_b": 1.0},
                "parse_status": "retry_ok",
            }
        ]

        repaired, replaced, unresolved = merge_repaired_rows(required, original, retry)

        self.assertEqual([row["id"] for row in repaired], required)
        self.assertEqual(repaired[0]["pred_label"], "A>B")
        self.assertEqual(repaired[1]["pred_label"], "B>A")
        self.assertEqual(replaced, ["failed"])
        self.assertEqual(unresolved, [])

    def test_merge_repaired_rows_reports_unresolved_retry_failure(self) -> None:
        required = ["ok", "failed"]
        original = [
            {
                "id": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            },
            {
                "id": "failed",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
            },
        ]
        retry = [
            {
                "id": "failed",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
            }
        ]

        repaired, replaced, unresolved = merge_repaired_rows(required, original, retry)
        report = coverage_report(required, repaired)

        self.assertEqual([row["id"] for row in repaired], ["ok"])
        self.assertEqual(replaced, [])
        self.assertEqual(unresolved, ["failed"])
        self.assertFalse(report["coverage_passed"])
        self.assertEqual(report["missing_pairwise_rows"], 1)

    def test_reparse_retry_rows_recovers_clear_natural_language_output(self) -> None:
        rows = [
            {
                "id": "retry-natural",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
                "raw_output": "Response A contains several errors and misconceptions.",
            }
        ]

        reparsed = reparse_retry_rows(rows, max_new_tokens=512)

        self.assertEqual(reparsed[0]["pred_label"], "B>A")
        self.assertEqual(reparsed[0]["parsed_scores"], {"score_a": 0.0, "score_b": 1.0})
        self.assertEqual(reparsed[0]["parse_status"], "retry_reparse_ok")


if __name__ == "__main__":
    unittest.main()
