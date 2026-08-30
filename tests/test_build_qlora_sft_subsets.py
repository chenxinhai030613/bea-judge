import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_qlora_sft_subsets import build_subsets  # noqa: E402


def row(idx: int, label: str, dataset: str) -> dict:
    return {
        "id": f"id{idx}",
        "dataset": dataset,
        "target_text": label,
        "prompt_text": "p",
    }


class BuildQloraSftSubsetsTest(unittest.TestCase):
    def test_build_subsets_writes_deterministic_stratified_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "src"
            source.mkdir()
            train_rows = [
                row(1, "[RESULT] A", "d1"),
                row(2, "[RESULT] A", "d1"),
                row(3, "[RESULT] B", "d1"),
                row(4, "[RESULT] B", "d1"),
                row(5, "[RESULT] Tie", "d2"),
                row(6, "[RESULT] Tie", "d2"),
                row(7, "[RESULT] A", "d2"),
                row(8, "[RESULT] B", "d2"),
            ]
            dev_rows = [row(101, "[RESULT] A", "d1"), row(102, "[RESULT] Tie", "d2")]
            (source / "train.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in train_rows),
                encoding="utf-8",
            )
            (source / "dev.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in dev_rows),
                encoding="utf-8",
            )

            summary = build_subsets(source_dir=source, output_root=tmp_path, sample_sizes=[4], seed=42)

            subset_dir = tmp_path / "m_prometheus_pairwise_sft50"
            subset_rows = [
                json.loads(line)
                for line in (subset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(subset_rows), 4)
            self.assertTrue((subset_dir / "metadata.json").exists())
            self.assertEqual(summary["subsets"][0]["tag"], "sft50")

            summary_2 = build_subsets(source_dir=source, output_root=tmp_path, sample_sizes=[4], seed=42)
            subset_rows_2 = [
                json.loads(line)
                for line in (subset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(subset_rows, subset_rows_2)
            self.assertEqual(summary["subsets"][0]["train"]["target_distribution"], summary_2["subsets"][0]["train"]["target_distribution"])


if __name__ == "__main__":
    unittest.main()
