import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qlora_next_experiments.json"


class QloraTrainingAblationMatrixTest(unittest.TestCase):
    def test_training_ablation_contains_epoch_and_sft_size_experiments(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        group = next(group for group in config["experiment_groups"] if group["id"] == "qlora_training_ablation")
        experiments = {item["id"]: item for item in group["experiments"]}

        self.assertIn("epoch_ablation", experiments)
        self.assertIn("sft_size_ablation", experiments)
        self.assertIn("run_qlora_ablation_grid.sh", experiments["epoch_ablation"]["command"])
        self.assertIn("build_qlora_sft_subsets.py", experiments["sft_size_ablation"]["precondition_command"])
        self.assertIn("summarize_qlora_ablation_grid.py", experiments["epoch_ablation"]["postprocess_command"])
        self.assertIn("run_qlora_ablation_grid.sh", experiments["sft_size_ablation"]["command"])


if __name__ == "__main__":
    unittest.main()
