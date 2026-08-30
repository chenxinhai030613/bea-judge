import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_expansion_builder import (  # noqa: E402
    SourceSpec,
    adapt_preference_pairs,
    adapt_helpsteer2,
    adapt_ragtruth,
    build_manifest,
    build_expansion,
    license_gate,
    pairwise_label_from_text,
    read_records,
)


class DatasetExpansionBuilderTest(unittest.TestCase):
    def test_license_gate_rejects_unknown_and_external_eval_sources(self):
        unknown = SourceSpec(
            key="bad",
            dataset="bad",
            task_family="open_qa",
            filename="bad.jsonl",
            url="https://example.com/bad",
            license="unknown",
            version="main",
            target=10,
        )
        external = SourceSpec(
            key="rewardbench",
            dataset="rewardbench_external_eval",
            task_family="external_eval",
            filename="rewardbench.jsonl",
            url="https://example.com/rewardbench",
            license="mixed-subset-license",
            version="main",
            target=10,
            redistribution_allowed=False,
            external_eval_only=True,
        )

        self.assertEqual(license_gate(unknown), (False, "license_missing_or_restricted"))
        self.assertEqual(license_gate(external), (False, "external_eval_only"))

    def test_adapt_helpsteer2_builds_pairwise_flat_samples(self):
        spec = SourceSpec(
            key="helpsteer2",
            dataset="helpsteer2",
            task_family="open_qa",
            filename="helpsteer2.jsonl",
            url="https://example.com/helpsteer2",
            license="CC-BY-4.0",
            version="main",
            target=5,
        )
        records = [
            {"id": "a", "prompt": "Explain photosynthesis.", "response": "Weak answer.", "overall": 1},
            {"id": "b", "prompt": "Explain photosynthesis.", "response": "Strong answer with details.", "overall": 5},
        ]

        samples = adapt_helpsteer2(records, spec, limit=5)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["dataset"], "helpsteer2")
        self.assertEqual(samples[0]["task_type"], "open_qa")
        self.assertEqual(samples[0]["human_label"], "A>B")
        self.assertEqual(samples[0]["human_score"]["pairwise_preference"], 1.0)

    def test_read_records_preserves_json_string_line_separators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(json.dumps({"prompt": "a\u2028b", "answer": "ok"}) + "\n", encoding="utf-8")

            rows = read_records(path)

            self.assertEqual(rows[0]["prompt"], "a\u2028b")

    def test_adapt_ragtruth_requires_context_and_maps_hallucination(self):
        spec = SourceSpec(
            key="ragtruth",
            dataset="ragtruth",
            task_family="factuality_rag",
            filename="ragtruth.jsonl",
            url="https://example.com/ragtruth",
            license="MIT",
            version="main",
            target=5,
        )
        records = [
            {
                "id": "r1",
                "question": "Who wrote Hamlet?",
                "context": "Hamlet is a tragedy written by William Shakespeare.",
                "response": "Hamlet was written by William Shakespeare.",
                "hallucination": 0,
            }
        ]

        samples = adapt_ragtruth(records, spec, limit=5)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["task_type"], "factuality_rag")
        self.assertEqual(samples[0]["human_label"], "supported")
        self.assertTrue(samples[0]["context"])
        self.assertEqual(samples[0]["metadata"]["label_scope"], "response_level_hallucination")
        self.assertEqual(samples[0]["metadata"]["source_label_schema"], "ragtruth_binary")
        self.assertEqual(samples[0]["metadata"]["original_ragtruth_record"]["id"], "r1")

    def test_pairwise_numeric_labels_do_not_use_factuality_mapping(self):
        self.assertEqual(pairwise_label_from_text(1), "A>B")
        self.assertEqual(pairwise_label_from_text("2"), "B>A")
        self.assertEqual(pairwise_label_from_text("tie"), "Tie")

    def test_adapt_preference_pairs_maps_offsetbias_numeric_labels(self):
        spec = SourceSpec(
            key="offsetbias",
            dataset="offsetbias",
            task_family="pairwise_bias",
            filename="offsetbias.jsonl",
            url="https://example.com/offsetbias",
            license="BSD-3-Clause",
            version="main",
            target=5,
        )
        rows = [
            {
                "id": "o1",
                "instruction": "Which response is less position-biased?",
                "output_1": "Response one.",
                "output_2": "Response two.",
                "label": 2,
            }
        ]

        samples = adapt_preference_pairs(rows, spec, limit=5)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["human_label"], "B>A")
        self.assertEqual(samples[0]["human_score"]["pairwise_preference"], -1.0)

    def test_build_manifest_prefers_source_metadata_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            metadata = {
                "sources": {
                    "helpsteer2": {
                        "acquisition_date": "2026-05-18T00:00:00+00:00",
                        "license_url": "https://example.com/license",
                        "revision": "abc123",
                        "sha256": "f" * 64,
                        "record_count": 12,
                    }
                }
            }
            (source_dir / "source_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            manifest = build_manifest(source_dir, [])
            item = manifest["sources"]["helpsteer2"]

            self.assertEqual(item["acquisition_date"], "2026-05-18T00:00:00+00:00")
            self.assertEqual(item["license_url"], "https://example.com/license")
            self.assertEqual(item["version"], "abc123")
            self.assertEqual(item["sha256"], "f" * 64)
            self.assertEqual(item["record_count"], 12)

    def test_build_expansion_writes_v2_outputs_when_minimum_count_met(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing_path = tmp / "existing.json"
            source_dir = tmp / "raw_v2"
            source_dir.mkdir()
            existing_samples = [
                {
                    "id": "legacy_1",
                    "dataset": "mt_bench",
                    "task_type": "open_qa",
                    "prompt": "Legacy prompt?",
                    "context": "",
                    "answer_a": "Legacy answer A.",
                    "answer_b": "Legacy answer B.",
                    "reference": "",
                    "human_score": {
                        "score_format": "pairwise_preference",
                        "scoring_system": "pairwise_preference",
                        "label": "A>B",
                        "pairwise_preference": 1.0,
                    },
                    "human_label": "A>B",
                    "language": "en",
                    "split": "train",
                    "metadata": {},
                }
            ]
            existing_path.write_text(json.dumps({"samples": existing_samples}), encoding="utf-8")
            (source_dir / "helpsteer2.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "h1", "prompt": "Prompt one?", "response": "Bad.", "overall": 1}),
                        json.dumps({"id": "h2", "prompt": "Prompt one?", "response": "Good detailed response.", "overall": 5}),
                    ]
                ),
                encoding="utf-8",
            )
            report = build_expansion(
                existing_path=existing_path,
                source_dir=source_dir,
                output_path=tmp / "processed.json",
                split_dir=tmp / "splits",
                manifest_path=tmp / "manifest.json",
                report_path=tmp / "report.json",
                data_availability_path=tmp / "da.md",
                target_count=2,
                minimum_count=2,
                allow_incomplete=False,
            )

            self.assertEqual(report["statistics"]["total_samples"], 2)
            self.assertTrue((tmp / "processed.json").exists())
            self.assertTrue((tmp / "splits" / "train.json").exists())
            self.assertTrue(report["gates"]["minimum_count_gate"])
            self.assertTrue(report["gates"]["factuality_context_missing_zero"])


if __name__ == "__main__":
    unittest.main()
