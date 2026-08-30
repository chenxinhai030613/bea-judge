import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_qlora_next_experiment_plan import build_markdown, main  # noqa: E402


class BuildQloraNextExperimentPlanTest(unittest.TestCase):
    def test_build_markdown_includes_protocol_policy_and_commands(self) -> None:
        config = json.loads((ROOT / "configs" / "qlora_next_experiments.json").read_text(encoding="utf-8"))

        markdown = build_markdown(config)

        self.assertIn("# QLoRA-BEA-Judge Next Experiment Plan", markdown)
        self.assertIn("accuracy-oriented operating point", markdown)
        self.assertIn("tie-sensitive operating point", markdown)
        self.assertIn("scripts/run_qlora_3seed_epoch1.sh", markdown)
        self.assertIn("four_module_ablation", markdown)
        self.assertIn("summarize_qlora_ablation_3seed.py", markdown)
        self.assertIn("epoch_ablation", markdown)
        self.assertIn("sft_size_ablation", markdown)
        self.assertIn("summarize_qlora_ablation_grid.py", markdown)
        self.assertIn("run_qlora_ablation_grid.sh", markdown)


    def test_main_writes_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.md"
            argv = [
                "build_qlora_next_experiment_plan.py",
                "--config",
                str(ROOT / "configs" / "qlora_next_experiments.json"),
                "--output",
                str(output),
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                main()
            finally:
                sys.argv = old_argv

            self.assertIn("Experiment Matrix", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
