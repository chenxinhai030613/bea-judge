import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qlora_next_experiments.json"


class QloraNextExperimentsConfigTest(unittest.TestCase):
    def test_experiment_matrix_covers_required_groups(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        groups = {group["id"]: group for group in config["experiment_groups"]}

        self.assertEqual(config["stable_protocol"]["seeds"], ["13", "42", "2026"])
        self.assertIn("main_comparison", groups)
        self.assertIn("qlora_training_ablation", groups)
        self.assertIn("four_module_ablation", groups)
        self.assertIn("tie_policy_robustness", groups)
        self.assertIn("calibration_ablation", groups)
        self.assertIn("stress_and_error_analysis", groups)

    def test_every_experiment_has_a_command_metrics_and_gate(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))

        for group in config["experiment_groups"]:
            self.assertIn("purpose", group)
            self.assertGreater(len(group["experiments"]), 0)
            for experiment in group["experiments"]:
                with self.subTest(group=group["id"], experiment=experiment["id"]):
                    self.assertIn("command", experiment)
                    self.assertGreater(len(experiment["metrics"]), 0)
                    self.assertGreater(len(experiment["gate"]), 0)

    def test_four_module_ablation_names_all_core_modules(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        group = next(group for group in config["experiment_groups"] if group["id"] == "four_module_ablation")
        variants = set()
        for experiment in group["experiments"]:
            variants.update(experiment.get("variants", []))

        self.assertIn("w/o Base Judge Scores", variants)
        self.assertIn("w/o Bias Module", variants)
        self.assertIn("w/o Evidence Module", variants)
        self.assertIn("w/o Calibration", variants)


if __name__ == "__main__":
    unittest.main()
