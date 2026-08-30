import json
import sys
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_qlora_ablation_3seed import build_summary, load_seed_reports, markdown_summary, write_summary  # noqa: E402
from summarize_qlora_3seed import comparison_dirs_from_args, output_dir_from_args  # noqa: E402


def fake_report(accuracy, macro_f1, tie_recall):
    return {
        "variants": [
            {
                "name": "Full BEA-Judge",
                "pairwise": {
                    "test_metrics": {
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "ece": 0.02,
                        "tie_recall": tie_recall,
                    }
                },
            },
            {
                "name": "w/o Bias Module",
                "pairwise": {
                    "test_metrics": {
                        "accuracy": accuracy - 0.01,
                        "macro_f1": macro_f1 - 0.02,
                        "ece": 0.03,
                        "tie_recall": tie_recall,
                    }
                },
            },
        ],
        "feature_group_ablations": [
            {
                "name": "+weighted calibration",
                "test_metrics": {
                    "accuracy": 0.80,
                    "macro_f1": 0.75,
                    "ece": 0.01,
                },
            }
        ],
        "bias_utility": [
            {
                "setting": "bias_as_decision_features",
                "head": "pairwise",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "review_capture_rate": 0.7,
            }
        ],
    }


class SummarizeQloraAblation3SeedTest(unittest.TestCase):
    def test_build_summary_aggregates_variant_metrics(self) -> None:
        summary = build_summary(
            [
                {"seed": "13", "path": "seed13.json", "report": fake_report(0.80, 0.71, 0.45)},
                {"seed": "42", "path": "seed42.json", "report": fake_report(0.82, 0.73, 0.55)},
            ]
        )

        full_pairwise = summary["sections"]["variants"]["Full BEA-Judge"]["pairwise"]["metrics"]
        self.assertEqual(full_pairwise["accuracy"]["n"], 2)
        self.assertEqual(full_pairwise["accuracy"]["mean"], 0.81)
        self.assertEqual(full_pairwise["tie_recall"]["mean"], 0.5)
        self.assertIn("feature_group_ablations", summary["sections"])
        self.assertIn("bias_utility", summary["sections"])

    def test_load_seed_reports_respects_allow_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "seed13.json").write_text(json.dumps(fake_report(0.80, 0.71, 0.45)), encoding="utf-8")

            reports = load_seed_reports(
                seeds=["13", "42"],
                report_template=str(tmp_path / "seed{seed}.json"),
                allow_missing=True,
            )

            self.assertEqual([report["seed"] for report in reports], ["13"])

    def test_write_summary_writes_json_and_markdown(self) -> None:
        summary = build_summary(
            [{"seed": "13", "path": "seed13.json", "report": fake_report(0.80, 0.71, 0.45)}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            write_summary(summary, output_dir)

            self.assertTrue((output_dir / "ablation_3seed_summary.json").exists())
            self.assertIn("Full BEA-Judge", (output_dir / "ablation_3seed_summary.md").read_text(encoding="utf-8"))
            self.assertIn("variants", markdown_summary(summary))

    def test_comparison_dirs_default_to_1024_suffix_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preferred = root / "qlora_comparison_seed13_epoch1_1024"
            legacy = root / "qlora_comparison_seed13_epoch1"
            preferred.mkdir(parents=True)
            legacy.mkdir(parents=True)

            args = SimpleNamespace(
                comparison_dirs=None,
                input_root=str(root),
                seeds=["13"],
                run_suffix="_1024",
            )

            dirs = comparison_dirs_from_args(args)

            self.assertEqual(dirs, [preferred])

    def test_output_dir_defaults_to_run_suffix(self) -> None:
        args = SimpleNamespace(output_dir=None, run_suffix="_1024")

        path = output_dir_from_args(args)

        self.assertEqual(
            path,
            ROOT / "datasets" / "model_outputs" / "qlora_3seed_epoch1_1024_summary",
        )

    def test_makefile_summarize_target_writes_to_tmp_by_default(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn(
            "QLORA_SUMMARY_OUTPUT ?= /tmp/bea-judge-qlora-3seed-summary",
            makefile,
        )
        self.assertIn("--output-dir $(QLORA_SUMMARY_OUTPUT)", makefile)


if __name__ == "__main__":
    unittest.main()
