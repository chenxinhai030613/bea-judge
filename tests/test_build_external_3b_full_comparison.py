import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_external_3b_full_comparison import build_summary, metric_cell  # noqa: E402


def metric(mean, std=0.0, n=3):
    return {"mean": mean, "std": std, "n": n}


class BuildExternal3BFullComparisonTest(unittest.TestCase):
    def test_metric_cell_formats_single_run_and_mean_std(self) -> None:
        self.assertEqual(metric_cell({"mean": 0.731244, "std": None}), "0.7312")
        self.assertEqual(metric_cell({"mean": 0.829693, "std": 0.003053}), "0.8297 +/- 0.0031")
        self.assertEqual(metric_cell({"mean": None, "std": None}), "")

    def test_build_summary_combines_internal_external_and_tie_rescue_rows(self) -> None:
        epoch_summary = {
            "results": [
                {
                    "setting": "epoch2_1024",
                    "by_system": {
                        "Raw M-Prometheus-3B": {
                            "accuracy": metric(0.56),
                            "macro_f1": metric(0.40),
                            "ece": metric(None, None, 0),
                            "tie_recall": metric(None, None, 0),
                        },
                        "Current BEA-Judge": {
                            "accuracy": metric(0.75),
                            "macro_f1": metric(0.67),
                            "ece": metric(0.05),
                            "tie_recall": metric(0.52),
                        },
                        "QLoRA-M-Prometheus-3B": {
                            "accuracy": metric(0.83),
                            "macro_f1": metric(0.72),
                            "ece": metric(0.13),
                            "tie_recall": metric(0.34),
                        },
                        "QLoRA-BEA-Judge": {
                            "accuracy": metric(0.8297, 0.0031),
                            "macro_f1": metric(0.7348, 0.0062),
                            "ece": metric(0.0278, 0.0031),
                            "tie_recall": metric(0.4256, 0.0270),
                        },
                    },
                }
            ]
        }
        external_report = {
            "baselines": {
                "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft": {
                    "test_metrics": {
                        "n": 1053,
                        "accuracy": 0.731244,
                        "macro_f1": 0.658361,
                        "ece": 0.1759,
                        "tie_recall": 0.5,
                        "parse_failure_rate": 0.0,
                    }
                },
                "Qwen/Qwen2.5-3B-Instruct": {
                    "test_metrics": {
                        "n": 1053,
                        "accuracy": 0.573599,
                        "macro_f1": 0.416005,
                        "ece": 0.347879,
                        "tie_recall": 0.030769,
                        "parse_failure_rate": 0.0,
                    }
                },
                "PatronusAI/glider": {
                    "test_metrics": {
                        "n": 1053,
                        "accuracy": 0.681862,
                        "macro_f1": 0.612345,
                        "ece": 0.201234,
                        "tie_recall": 0.423077,
                        "parse_failure_rate": 0.0,
                    }
                },
            }
        }
        tie_rescue_audit = {
            "results": [
                {
                    "setting": "epoch2_1024",
                    "mean_std": {
                        "test": {
                            "accuracy": {"mean": 0.829693, "std": 0.005231},
                            "macro_f1": {"mean": 0.744096, "std": 0.00926},
                            "ece": {"mean": 0.028308, "std": 0.002723},
                            "tie_recall": {"mean": 0.479487, "std": 0.048852},
                        }
                    },
                }
            ]
        }

        summary = build_summary(
            epoch_summary=epoch_summary,
            external_report=external_report,
            tie_rescue_audit=tie_rescue_audit,
        )

        self.assertEqual(len(summary["rows"]), 8)
        self.assertEqual(summary["rows"][3]["system"], "GRM-Llama3.2-3B reward model")
        self.assertEqual(summary["rows"][5]["system"], "GLIDER")
        self.assertEqual(summary["rows"][5]["n"], 1053)
        self.assertEqual(summary["rows"][5]["parse_failure_rate"], 0.0)
        self.assertNotIn("Prometheus-2 7B", [row["system"] for row in summary["rows"]])
        self.assertAlmostEqual(summary["key_deltas"]["qlora_bea_epoch2_minus_grm"]["accuracy"], 0.098456)
        self.assertAlmostEqual(summary["key_deltas"]["qlora_bea_epoch2_minus_glider"]["accuracy"], 0.147838)
        self.assertGreater(summary["key_deltas"]["tie_rescue_minus_qlora_bea_epoch2"]["tie_recall"], 0)


if __name__ == "__main__":
    unittest.main()
