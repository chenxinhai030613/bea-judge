import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qlora_judge_train import encode_sft_example  # noqa: E402


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class QloraJudgeTrainTest(unittest.TestCase):
    def test_encode_sft_example_masks_prompt_and_keeps_target(self) -> None:
        tokenizer = FakeTokenizer()

        encoded = encode_sft_example(
            prompt_text="abcdef",
            target_text="[RESULT] A",
            tokenizer=tokenizer,
            max_length=18,
        )

        target_ids = tokenizer("[RESULT] A<eos>")["input_ids"]
        self.assertLessEqual(len(encoded["input_ids"]), 18)
        self.assertEqual(encoded["input_ids"][-len(target_ids) :], target_ids)
        self.assertEqual(encoded["labels"][-len(target_ids) :], target_ids)
        self.assertTrue(all(value == -100 for value in encoded["labels"][: -len(target_ids)]))


if __name__ == "__main__":
    unittest.main()
