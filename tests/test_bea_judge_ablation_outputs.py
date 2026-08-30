import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bea_judge_ablation import write_report  # noqa: E402


class BeaJudgeAblationOutputsTest(unittest.TestCase):
    def test_write_report_accepts_custom_output_paths(self) -> None:
        report = {
            "created_at": "2026-05-28T00:00:00+00:00",
            "input_dataset": "datasets/processed/unit.json",
            "judge_output_path": "datasets/judge_outputs/unit.json",
            "local_prototype": False,
            "variants": [
                {
                    "name": "Full BEA-Judge",
                    "pairwise": {"test_metrics": {"accuracy": 1.0, "macro_f1": 1.0}},
                    "factuality": {"test_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0}},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_json = Path(tmp) / "seed13" / "ablation_report.json"
            report_md = Path(tmp) / "seed13" / "ablation_report.md"

            write_report(report, report_json=report_json, report_md=report_md)

            self.assertEqual(json.loads(report_json.read_text(encoding="utf-8"))["variants"][0]["name"], "Full BEA-Judge")
            self.assertIn("# BEA-Judge Ablation Report", report_md.read_text(encoding="utf-8"))

    def test_reuse_model_cli_flag_is_documented_in_experiment_matrix(self) -> None:
        matrix = json.loads((ROOT / "configs" / "qlora_next_experiments.json").read_text(encoding="utf-8"))
        commands = [
            experiment["command"]
            for group in matrix["experiment_groups"]
            for experiment in group["experiments"]
        ]

        self.assertTrue(any("--reuse-model-only" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
