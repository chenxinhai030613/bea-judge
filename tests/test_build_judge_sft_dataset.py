import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_judge_sft_dataset import build_dataset  # noqa: E402


class BuildJudgeSftDatasetTest(unittest.TestCase):
    def test_exports_only_train_and_dev_pairwise_rows(self) -> None:
        payload = {
            "samples": [
                {
                    "id": "train-a",
                    "split": "train",
                    "dataset": "unit",
                    "task_type": "open_qa",
                    "human_label": "A>B",
                    "prompt": "Prompt",
                    "answer_a": "A",
                    "answer_b": "B",
                },
                {
                    "id": "dev-tie",
                    "split": "dev",
                    "dataset": "unit",
                    "task_type": "open_qa",
                    "human_label": "Tie",
                    "prompt": "Prompt",
                    "answer_a": "A",
                    "answer_b": "B",
                },
                {
                    "id": "test-b",
                    "split": "test",
                    "dataset": "unit",
                    "task_type": "open_qa",
                    "human_label": "B>A",
                    "prompt": "Prompt",
                    "answer_a": "A",
                    "answer_b": "B",
                },
                {
                    "id": "single",
                    "split": "train",
                    "dataset": "unit",
                    "task_type": "factuality_rag",
                    "human_label": "supported",
                    "prompt": "Prompt",
                    "answer_a": "A",
                    "answer_b": None,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.json"
            output_dir = tmp / "sft"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            metadata = build_dataset(input_path=input_path, output_dir=output_dir)
            train_rows = [
                json.loads(line)
                for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dev_rows = [
                json.loads(line)
                for line in (output_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(metadata["train_rows"], 1)
        self.assertEqual(metadata["dev_rows"], 1)
        self.assertEqual(metadata["heldout_test_pairwise_rows"], 1)
        self.assertEqual(train_rows[0]["id"], "train-a")
        self.assertEqual(train_rows[0]["target_text"], "[RESULT] A")
        self.assertEqual(dev_rows[0]["id"], "dev-tie")
        self.assertEqual(dev_rows[0]["target_text"], "[RESULT] Tie")
        self.assertNotIn("test-b", {row["id"] for row in train_rows + dev_rows})
        self.assertIn("###Response A:", train_rows[0]["prompt_text"])


if __name__ == "__main__":
    unittest.main()
