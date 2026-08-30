import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qlora_judge_train import load_jsonl  # noqa: E402


class QloraJudgeTrainSftDirTest(unittest.TestCase):
    def test_load_jsonl_reads_subset_dir_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            rows = [
                {"prompt_text": "p1", "target_text": "[RESULT] A"},
                {"prompt_text": "p2", "target_text": "[RESULT] Tie"},
            ]
            train_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            loaded = load_jsonl(train_path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[1]["target_text"], "[RESULT] Tie")


if __name__ == "__main__":
    unittest.main()
