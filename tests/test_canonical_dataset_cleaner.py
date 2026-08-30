import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_dataset_cleaner import transform_sample, validate_canonical_sample


class CanonicalDatasetCleanerTest(unittest.TestCase):
    def test_transforms_flat_pairwise_sample_to_nested_schema(self) -> None:
        sample = {
            "id": "unit-1",
            "dataset": "wikieval_grounded_vs_poor",
            "task_type": "open_qa",
            "prompt": "Explain calibration.",
            "context": "",
            "answer_a": "A calibrated model aligns confidence with observed accuracy.",
            "answer_b": "Calibration is good.",
            "reference": "",
            "human_score": {
                "scoring_system": "pairwise_preference",
                "pairwise_preference": 1.0,
                "label": "A>B",
            },
            "human_label": "A>B",
            "language": "en",
            "split": "train",
            "metadata": {
                "source": "RAGAS WikiEval",
                "source_url": "https://example.test/wikieval",
                "original_index": 7,
                "scoring_system": "pairwise_preference",
            },
        }

        result = transform_sample(sample, "train", ["https://fallback.test/source"])
        errors, warnings = validate_canonical_sample(result.record)

        self.assertEqual(errors, [])
        self.assertIn("optional_field_normalized_to_null:input.context", result.warnings)
        self.assertIn("optional_field_normalized_to_null:input.reference", result.warnings)
        self.assertEqual(warnings, [])
        self.assertEqual(set(result.record), {"id", "source", "task", "input", "answers", "label", "quality", "metadata"})
        self.assertEqual(result.record["source"]["dataset"], "wikieval")
        self.assertEqual(result.record["source"]["source_record_id"], "7")
        self.assertIsNone(result.record["input"]["context"])
        self.assertIsNone(result.record["input"]["reference"])
        self.assertEqual(result.record["task"]["form"], "pairwise")
        self.assertEqual(result.record["label"]["score"], 1.0)

    def test_single_answer_placeholder_becomes_null_answer_b(self) -> None:
        sample = {
            "id": "unit-2",
            "dataset": "ares_nq",
            "task_type": "factuality_rag",
            "prompt": "Is the answer supported?",
            "context": "The cited document supports the answer.",
            "answer_a": "The answer is supported by the document.",
            "answer_b": "[SINGLE_ANSWER_FACTUALITY_TASK]",
            "reference": "Supported.",
            "human_score": {
                "scoring_system": "single_answer_factuality",
                "factuality_label_score": 1.0,
                "label": "supported",
            },
            "human_label": "supported",
            "language": "en",
            "split": "test",
            "metadata": {
                "source": "stanford-futuredata/ARES nq_labeled_output.tsv",
                "source_url": "https://example.test/ares",
                "row_id": "42",
                "factuality_task_form": "single_answer",
            },
        }

        result = transform_sample(sample, "test", ["https://fallback.test/source"])
        errors, warnings = validate_canonical_sample(result.record)

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertIsNone(result.record["answers"]["b"])
        self.assertFalse(result.record["quality"]["answer_b_required"])
        self.assertEqual(result.record["quality"]["missing_reason"]["answer_b"], "single_answer_task")
        self.assertEqual(result.record["label"]["type"], "single_answer_factuality")
        self.assertEqual(result.record["label"]["score"], 1.0)

    def test_chinese_dataset_name_is_normalized(self) -> None:
        sample = {
            "id": "zh-1",
            "dataset": "zh_professional_open_qa",
            "task_type": "open_qa",
            "prompt": "请给出项目风险控制方案。",
            "context": None,
            "answer_a": "先识别风险，再设置责任人与监测指标。",
            "answer_b": "只需要开会讨论。",
            "reference": None,
            "human_score": {
                "scoring_system": "pairwise_preference",
                "pairwise_preference": -1.0,
                "label": "B>A",
            },
            "human_label": "B>A",
            "language": "zh",
            "split": "dev",
            "metadata": {"source": "self_built_chinese_annotation"},
        }

        result = transform_sample(sample, "dev", ["self_built_chinese_annotation"])
        errors, _warnings = validate_canonical_sample(result.record)

        self.assertEqual(errors, [])
        self.assertEqual(result.record["source"]["dataset"], "zh_professional")
        self.assertIsNone(result.record["source"]["source_url"])
        self.assertIn("source_record_id_null", result.warnings)
        self.assertEqual(result.record["quality"]["missing_reason"]["context"], "not_required_for_task")

    def test_v2_source_names_are_valid_canonical_datasets(self) -> None:
        for dataset in ("helpsteer2", "offsetbias", "oasst1"):
            sample = {
                "id": f"{dataset}-1",
                "dataset": dataset,
                "task_type": "open_qa",
                "prompt": "Explain calibration.",
                "context": "",
                "answer_a": "A calibrated model aligns confidence and accuracy.",
                "answer_b": "Calibration is unrelated.",
                "reference": "",
                "human_score": {
                    "scoring_system": "pairwise_preference",
                    "pairwise_preference": 1.0,
                    "label": "A>B",
                },
                "human_label": "A>B",
                "language": "en",
                "split": "train",
                "metadata": {"source_url": "https://example.test/source", "source_record_id": "1"},
            }

            result = transform_sample(sample, "train", ["https://example.test/source"])
            errors, _warnings = validate_canonical_sample(result.record)

            self.assertNotIn("source.dataset_invalid", errors)

    def test_ragtruth_is_valid_factuality_source(self) -> None:
        sample = {
            "id": "ragtruth-1",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "prompt": "Is the answer supported?",
            "context": "The passage states that water freezes at 0 degrees Celsius.",
            "answer_a": "Water freezes at 0 degrees Celsius.",
            "answer_b": "[SINGLE_ANSWER_FACTUALITY_TASK]",
            "reference": "",
            "human_score": {
                "scoring_system": "single_answer_factuality",
                "factuality_label_score": 1.0,
                "label": "supported",
            },
            "human_label": "supported",
            "language": "en",
            "split": "train",
            "metadata": {
                "source_url": "https://example.test/ragtruth",
                "source_record_id": "r1",
                "factuality_task_form": "single_answer",
            },
        }

        result = transform_sample(sample, "train", ["https://example.test/ragtruth"])
        errors, _warnings = validate_canonical_sample(result.record)

        self.assertNotIn("source.dataset_invalid", errors)
        self.assertEqual(result.record["source"]["dataset"], "ragtruth")


if __name__ == "__main__":
    unittest.main()
