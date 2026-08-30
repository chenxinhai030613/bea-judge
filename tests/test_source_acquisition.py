import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from source_acquisition import (  # noqa: E402
    acquire_sources,
    export_oasst1_pairs,
    response_label,
    row_score,
    selected_specs,
)


class SourceAcquisitionTest(unittest.TestCase):
    def test_selected_specs_default_excludes_rewardbench(self):
        specs = selected_specs(["training"], include_rewardbench=False)
        keys = {spec.key for spec in specs}

        self.assertIn("helpsteer2", keys)
        self.assertIn("ragtruth", keys)
        self.assertNotIn("rewardbench", keys)

    def test_selected_specs_can_include_rewardbench_as_external_eval(self):
        specs = selected_specs(["training"], include_rewardbench=True)
        rewardbench = [spec for spec in specs if spec.key == "rewardbench"][0]

        self.assertTrue(rewardbench.external_eval_only)
        self.assertFalse(rewardbench.training_admission)

    def test_metadata_only_records_existing_file_sha_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            raw = source_dir / "helpsteer2.jsonl"
            raw.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "p1", "response": "a1", "overall": 1}),
                        json.dumps({"prompt": "p1", "response": "a2", "overall": 5}),
                    ]
                ),
                encoding="utf-8",
            )
            metadata_path = source_dir / "source_metadata.json"

            report = acquire_sources(
                source_dir=source_dir,
                metadata_path=metadata_path,
                sources=["helpsteer2"],
                metadata_only=True,
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            item = payload["sources"]["helpsteer2"]

            self.assertEqual(report["reports"]["helpsteer2"]["record_count"], 2)
            self.assertEqual(item["record_count"], 2)
            self.assertTrue(item["sha256"])
            self.assertTrue(item["acquisition_date"])
            self.assertEqual(item["license"], "CC-BY-4.0")

    def test_export_oasst1_pairs_builds_ranked_preference_pair(self):
        rows = [
            {"message_id": "p1", "role": "prompter", "text": "Prompt?", "lang": "en"},
            {"message_id": "a1", "parent_id": "p1", "role": "assistant", "text": "Weak.", "rank": 1},
            {"message_id": "a2", "parent_id": "p1", "role": "assistant", "text": "Strong.", "rank": 0},
        ]

        pairs = list(export_oasst1_pairs(rows))

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], "Prompt?")
        self.assertEqual(pairs[0]["answer_a"], "Strong.")
        self.assertEqual(pairs[0]["answer_b"], "Weak.")
        self.assertEqual(pairs[0]["label"], "A>B")

    def test_row_score_prefers_scalar_score_then_oasst_labels(self):
        self.assertEqual(row_score({"helpfulness": 4}), 4.0)
        self.assertEqual(
            row_score({"labels": {"name": ["quality", "helpfulness"], "value": [0.5, 1.0]}}),
            0.75,
        )

    def test_response_label_maps_ragtruth_labels_to_factuality(self):
        self.assertEqual(response_label({"labels": [{"label_type": "Evident Conflict"}]}), "unsupported")
        self.assertEqual(response_label({"labels": [], "quality": "good"}), "supported")


if __name__ == "__main__":
    unittest.main()
