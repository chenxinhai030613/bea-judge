import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bea_judge_train import read_dataset
from dataset_adapter import adapt_sample, canonical_to_flat_sample, is_canonical_sample


class DatasetAdapterTest(unittest.TestCase):
    def test_canonical_to_flat_sample_preserves_existing_training_contract(self) -> None:
        canonical = {
            "id": "unit-1",
            "source": {
                "dataset": "mt_bench",
                "source_url": "https://example.test/mt",
                "source_record_id": "q-1",
            },
            "task": {
                "type": "open_qa",
                "form": "pairwise",
                "language": "en",
                "split": "train",
            },
            "input": {
                "prompt": "Explain calibration.",
                "context": None,
                "reference": None,
            },
            "answers": {
                "a": "Calibration aligns confidence with observed correctness.",
                "b": "Calibration is a setup step.",
            },
            "label": {
                "type": "pairwise_preference",
                "value": "A>B",
                "score": 1.0,
            },
            "quality": {
                "context_required": False,
                "reference_required": False,
                "answer_b_required": True,
                "missing_reason": {"context": "not_required_for_task"},
            },
            "metadata": {"source": "unit"},
        }

        flat = canonical_to_flat_sample(canonical)

        self.assertEqual(flat["id"], "unit-1")
        self.assertEqual(flat["dataset"], "mt_bench")
        self.assertEqual(flat["task_type"], "open_qa")
        self.assertEqual(flat["prompt"], "Explain calibration.")
        self.assertIsNone(flat["context"])
        self.assertEqual(flat["answer_a"], "Calibration aligns confidence with observed correctness.")
        self.assertEqual(flat["answer_b"], "Calibration is a setup step.")
        self.assertEqual(flat["human_label"], "A>B")
        self.assertEqual(flat["human_score"]["scoring_system"], "pairwise_preference")
        self.assertEqual(flat["human_score"]["pairwise_preference"], 1.0)
        self.assertEqual(flat["metadata"]["source_url"], "https://example.test/mt")
        self.assertEqual(flat["metadata"]["source_record_id"], "q-1")
        self.assertEqual(flat["metadata"]["canonical_quality"]["context_required"], False)

    def test_single_answer_form_sets_factuality_task_metadata(self) -> None:
        canonical = {
            "id": "unit-2",
            "source": {"dataset": "ares_nq", "source_url": None, "source_record_id": "42"},
            "task": {"type": "factuality_rag", "form": "single_answer", "language": "en", "split": "test"},
            "input": {"prompt": "Is this supported?", "context": "Evidence.", "reference": "Supported."},
            "answers": {"a": "It is supported.", "b": None},
            "label": {"type": "single_answer_factuality", "value": "supported", "score": 1.0},
            "quality": {"context_required": True, "reference_required": False, "answer_b_required": False, "missing_reason": {}},
            "metadata": {},
        }

        flat = canonical_to_flat_sample(canonical)

        self.assertIsNone(flat["answer_b"])
        self.assertEqual(flat["human_label"], "supported")
        self.assertEqual(flat["human_score"]["factuality_label_score"], 1.0)
        self.assertEqual(flat["human_score"]["factuality_score_0_1"], 1.0)
        self.assertEqual(flat["metadata"]["factuality_task_form"], "single_answer")

    def test_flat_sample_is_returned_without_conversion(self) -> None:
        flat = {
            "id": "flat-1",
            "dataset": "pandalm",
            "task_type": "open_qa",
            "prompt": "Prompt",
            "answer_a": "A",
            "answer_b": "B",
            "human_label": "Tie",
        }

        self.assertFalse(is_canonical_sample(flat))
        self.assertIs(adapt_sample(flat), flat)

    def test_training_reader_accepts_canonical_payload(self) -> None:
        canonical = {
            "id": "unit-3",
            "source": {"dataset": "zh_professional", "source_url": None, "source_record_id": None},
            "task": {"type": "open_qa", "form": "pairwise", "language": "zh", "split": "dev"},
            "input": {"prompt": "请解释校准。", "context": None, "reference": None},
            "answers": {"a": "校准用于让置信度接近真实正确率。", "b": "校准是一个步骤。"},
            "label": {"type": "pairwise_preference", "value": "A>B", "score": 1.0},
            "quality": {"context_required": False, "reference_required": False, "answer_b_required": True, "missing_reason": {}},
            "metadata": {},
        }
        payload = {"dataset_info": {"schema": "BEA-Judge canonical nested schema"}, "samples": [canonical]}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            samples = read_dataset(path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["dataset"], "zh_professional")
        self.assertEqual(samples[0]["task_type"], "open_qa")
        self.assertEqual(samples[0]["language"], "zh")


if __name__ == "__main__":
    unittest.main()
