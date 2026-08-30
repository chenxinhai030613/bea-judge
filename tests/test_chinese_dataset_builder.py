import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chinese_dataset_builder import DEFAULT_ZH_TARGET, build_chinese_samples
from dataset_builder import split_group_key


def blank_string_paths(value, path="$"):
    if isinstance(value, str):
        return [path] if not value.strip() else []
    if isinstance(value, dict):
        paths = []
        for key, child in value.items():
            paths.extend(blank_string_paths(child, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for idx, child in enumerate(value):
            paths.extend(blank_string_paths(child, f"{path}[{idx}]"))
        return paths
    return []


class ChineseDatasetBuilderTest(unittest.TestCase):
    def test_default_target_is_1000(self) -> None:
        self.assertEqual(DEFAULT_ZH_TARGET, 1000)

    def test_builds_1000_samples_with_contracts_and_no_blank_strings(self) -> None:
        samples, build_meta = build_chinese_samples(1000)

        self.assertEqual(len(samples), 1000)
        self.assertEqual(build_meta["selected_counts"], {"open_qa": 400, "pairwise_bias": 400, "factuality_rag": 200})
        self.assertEqual(Counter(sample["task_type"] for sample in samples), build_meta["selected_counts"])
        self.assertEqual(Counter(sample["language"] for sample in samples), {"zh": 1000})
        self.assertEqual(len({sample["id"] for sample in samples}), 1000)

        split_by_group = defaultdict(set)
        for sample in samples:
            split_by_group[split_group_key(sample)].add(sample["split"])
        leaked_groups = {key: splits for key, splits in split_by_group.items() if len(splits) > 1}
        self.assertEqual(leaked_groups, {})

        blank_paths = []
        for sample in samples:
            blank_paths.extend((sample["id"], path) for path in blank_string_paths(sample))
        self.assertEqual(blank_paths, [])

        for sample in samples:
            meta = sample["metadata"]
            self.assertEqual(meta["null_normalization"], "optional_empty_text_fields_use_null")
            self.assertIn("field_contract", meta)
            task_type = sample["task_type"]
            if task_type in {"open_qa", "pairwise_bias"}:
                self.assertIsNone(sample["context"])
                self.assertIsNone(sample["reference"])
                self.assertIsInstance(sample["answer_b"], str)
                self.assertEqual(meta["missing_reason"]["context"], "not_required_for_task")
                self.assertEqual(meta["missing_reason"]["reference"], "not_required_for_task")
            elif task_type == "factuality_rag":
                self.assertIsInstance(sample["context"], str)
                self.assertIsInstance(sample["reference"], str)
                self.assertIsInstance(sample["answer_b"], str)
            else:
                self.fail(f"unexpected task_type: {task_type}")


if __name__ == "__main__":
    unittest.main()
