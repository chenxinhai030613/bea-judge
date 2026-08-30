import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_sci_results_tables import (  # noqa: E402
    build_baseline_comparison_rows,
    build_base_diagnostic_rows,
    build_bias_subgroup_rows,
    build_bias_risk_utility_rows,
    build_calibration_method_rows,
    build_evidence_subtype_rows,
    build_expansion_distribution_rows,
    build_license_audit_rows,
    build_metric_ci_rows,
    build_per_dataset_rows,
    build_ragtruth_result_rows,
    build_risk_coverage_rows,
    build_source_provenance_rows,
    build_swap_consistency_rows,
    count_unresolved_rows,
    generate_outputs,
    metrics_for_rows,
    mcnemar_exact_or_chi2_pvalue,
    validate_sci_gates,
)


def pairwise_row(sample_id, dataset, gold, pred, confidence=0.8):
    probs = {"A>B": 0.1, "B>A": 0.1, "Tie": 0.1}
    probs[pred] = confidence
    return {
        "id": sample_id,
        "dataset": dataset,
        "task_type": "open_qa",
        "split": "test",
        "head": "pairwise",
        "human_label": gold,
        "predicted_label": pred,
        "confidence": confidence,
        "review_flag": confidence < 0.5,
        "label_probabilities": probs,
    }


def factuality_row(sample_id, dataset, gold, pred, confidence=0.8):
    probs = {"supported": 0.1, "unsupported": 0.1, "ambiguous": 0.0}
    probs[pred] = confidence
    return {
        "id": sample_id,
        "dataset": dataset,
        "task_type": "factuality_rag",
        "split": "test",
        "head": "factuality",
        "human_label": gold,
        "predicted_label": pred,
        "confidence": confidence,
        "review_flag": confidence < 0.5,
        "label_probabilities": probs,
    }


class GenerateSciResultsTablesTest(unittest.TestCase):
    def test_metrics_for_rows_reports_tie_recall(self):
        rows = [
            pairwise_row("s1", "mt_bench", "Tie", "Tie", 0.55),
            pairwise_row("s2", "mt_bench", "Tie", "A>B", 0.60),
            pairwise_row("s3", "mt_bench", "A>B", "A>B", 0.90),
        ]

        metrics = metrics_for_rows(rows, ["A>B", "B>A", "Tie"])

        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["accuracy"], 0.6667)
        self.assertEqual(metrics["tie_recall"], 0.5)
        self.assertIn("brier", metrics)

    def test_build_per_dataset_rows_groups_test_rows_by_head(self):
        payload = {
            "test": {
                "pairwise": [
                    pairwise_row("p1", "mt_bench", "A>B", "A>B"),
                    pairwise_row("p2", "judgebench", "Tie", "A>B"),
                ],
                "factuality": [
                    factuality_row("f1", "ares_nq", "supported", "supported"),
                ],
            }
        }

        rows = build_per_dataset_rows(payload)

        keys = {(row["head"], row["dataset"]) for row in rows}
        self.assertEqual(keys, {("pairwise", "judgebench"), ("pairwise", "mt_bench"), ("factuality", "ares_nq")})

    def test_build_baseline_comparison_rows_includes_control_baselines(self):
        report = {
            "variants": [
                {
                    "name": "Full BEA-Judge",
                    "pairwise": {
                        "test_metrics": {"accuracy": 0.8, "macro_f1": 0.7, "tie_recall": 0.5},
                        "test_rows": [pairwise_row("p1", "mt_bench", "A>B", "A>B")],
                    },
                }
            ],
            "control_baselines": [
                {
                    "name": "Raw M-Prometheus-3B only",
                    "pairwise": {
                        "test_metrics": {"accuracy": 0.5, "macro_f1": 0.4, "tie_recall": 0.1},
                        "test_rows": [pairwise_row("p1", "mt_bench", "A>B", "Tie")],
                    },
                }
            ],
        }

        rows = build_baseline_comparison_rows(report)
        by_name = {row["system"]: row for row in rows}

        self.assertEqual(by_name["Raw M-Prometheus-3B only"]["source"], "control")
        self.assertEqual(by_name["Full BEA-Judge"]["source"], "module_variant")

    def test_build_base_diagnostic_rows_compares_base_and_calibrated_predictions(self):
        base_scores = [
            {
                "id": "p1",
                "judge_backend": "m_prometheus",
                "parse_status": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            }
        ]
        calibrated = {"test": {"pairwise": [pairwise_row("p1", "mt_bench", "A>B", "Tie")]}}

        rows = build_base_diagnostic_rows(base_scores, calibrated)

        self.assertEqual(rows[0]["base_accuracy"], 1.0)
        self.assertEqual(rows[0]["calibrated_accuracy"], 0.0)
        self.assertEqual(rows[0]["base_conflict_rate"], 0.0)

    def test_build_bias_subgroup_rows_includes_required_groups(self):
        profiles = {
            "profiles": [
                {
                    "id": "p1",
                    "dataset": "unit",
                    "task_type": "pairwise_bias",
                    "split": "test",
                    "metadata": {"bias_type": "position"},
                    "prediction": {"predicted_label": "A>B", "gold_label": "A>B", "confidence": 0.8},
                    "bias": {"overall_bias_risk": 0.7, "review_required": True},
                }
            ]
        }

        rows = build_bias_subgroup_rows(profiles)
        groups = {row["bias_group"] for row in rows}

        self.assertIn("position", groups)
        self.assertIn("rubric_sensitivity", groups)

    def test_build_evidence_subtype_rows_reports_error_capture(self):
        profiles = {
            "profiles": [
                {
                    "id": "f1",
                    "evidence": {"evidence_risk": 0.9, "reasons": ["numeric_evidence_gap_a"]},
                }
            ]
        }
        calibrated = {"test": {"factuality": [factuality_row("f1", "ares_nq", "supported", "unsupported", 0.4)]}}

        rows = build_evidence_subtype_rows(profiles, calibrated)
        numeric = [row for row in rows if row["evidence_subtype"] == "numeric_evidence_gap"][0]

        self.assertEqual(numeric["error_count"], 1)
        self.assertEqual(numeric["review_capture_rate"], 1.0)

    def test_build_swap_consistency_rows_uses_probe_dataset_summary(self):
        report = {
            "dataset_summary": [
                {
                    "dataset": "overall",
                    "selected_n": 4,
                    "swap_available_n": 3,
                    "swap_parse_success_rate": 0.75,
                    "swap_consistency_rate": 0.6667,
                    "swap_inconsistency_rate": 0.3333,
                    "calibrated_error_rate": 0.5,
                    "error_rate_when_inconsistent": 1.0,
                    "avg_swap_margin_delta": 0.25,
                    "tie_case_rate": 0.5,
                }
            ]
        }

        rows = build_swap_consistency_rows(report)

        self.assertEqual(rows[0]["dataset"], "overall")
        self.assertEqual(rows[0]["swap_available_n"], 3)
        self.assertEqual(rows[0]["swap_consistency_rate"], 0.6667)

    def test_build_risk_coverage_rows_reports_error_capture(self):
        payload = {
            "test": {
                "pairwise": [
                    {**pairwise_row("p1", "mt_bench", "A>B", "B>A", 0.4), "risk_score": 0.6},
                    {**pairwise_row("p2", "mt_bench", "A>B", "A>B", 0.9), "risk_score": 0.1},
                ],
                "factuality": [],
            }
        }

        rows = build_risk_coverage_rows(payload)
        first = [row for row in rows if row["head"] == "pairwise" and row["review_rate"] == 0.5][0]

        self.assertEqual(first["error_capture_rate"], 1.0)
        self.assertEqual(first["auto_accept_accuracy"], 1.0)

    def test_build_bias_risk_utility_rows_passes_ablation_payload(self):
        report = {
            "bias_utility": [
                {
                    "setting": "bias_risk_only_review",
                    "head": "pairwise",
                    "split": "test",
                    "n": 2,
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "ece": 0.1,
                    "review_rate": 0.25,
                    "review_capture_rate": 1.0,
                }
            ]
        }

        rows = build_bias_risk_utility_rows(report)

        self.assertEqual(rows[0]["setting"], "bias_risk_only_review")
        self.assertEqual(rows[0]["review_capture_rate"], 1.0)

    def test_build_calibration_method_rows_flattens_summary(self):
        summary = {
            "results": {
                "temperature": {
                    "metrics_dev": {"accuracy": 0.8, "ece": 0.1, "mce": 0.2, "brier": 0.3, "nll": 0.4},
                    "metrics_test": {"accuracy": 0.7, "ece": 0.2, "mce": 0.3, "brier": 0.4, "nll": 0.5},
                    "extras": {},
                },
                "conformal": {
                    "metrics_dev": {"accuracy": 0.8, "ece": 0.1, "mce": 0.2, "brier": 0.3, "nll": 0.4},
                    "metrics_test": {"accuracy": 0.7, "ece": 0.2, "mce": 0.3, "brier": 0.4, "nll": 0.5},
                    "extras": {"coverage_dev": 0.91, "coverage_test": 0.9, "set_size_avg_test": 1.5},
                },
            }
        }

        rows = build_calibration_method_rows(summary)
        conformal_test = [row for row in rows if row["method"] == "conformal" and row["split"] == "test"][0]

        self.assertEqual(conformal_test["coverage"], 0.9)
        self.assertEqual(conformal_test["set_size_avg"], 1.5)

    def test_build_source_provenance_rows_combines_manifest_and_expansion_report(self):
        manifest = {
            "sources": {
                "helpsteer2": {
                    "license": "CC-BY-4.0",
                    "redistribution_allowed": True,
                    "admission_allowed": True,
                    "admission_reason": "ok",
                    "acquisition_date": "2026-05-18T00:00:00+00:00",
                    "sha256": "abc",
                }
            }
        }
        expansion_report = {"source_reports": [{"source": "helpsteer2", "accepted_records": 2}]}

        rows = build_source_provenance_rows(manifest, expansion_report)

        self.assertEqual(rows[0]["source"], "helpsteer2")
        self.assertEqual(rows[0]["accepted_records"], 2)
        self.assertTrue(rows[0]["sha256_present"])

    def test_build_expansion_distribution_rows_flattens_statistics(self):
        report = {
            "statistics": {
                "total_samples": 2,
                "by_task_type": {"open_qa": 2},
                "by_dataset": {"helpsteer2": 2},
                "by_split": {"train": 1, "test": 1},
                "by_language": {"en": 2},
                "human_label_distribution": {"A>B": 2},
            }
        }

        rows = build_expansion_distribution_rows(report)

        self.assertIn({"dimension": "by_task_type", "value": "open_qa", "count": 2}, rows)

    def test_build_license_audit_rows_flags_missing_acquisition_metadata(self):
        manifest = {
            "sources": {
                "helpsteer2": {
                    "license": "CC-BY-4.0",
                    "license_status": "present",
                    "redistribution_allowed": True,
                    "external_eval_only": False,
                    "admission_allowed": True,
                    "admission_reason": "ok",
                    "acquisition_date": None,
                    "sha256": None,
                },
                "rewardbench": {
                    "license": "mixed-subset-license",
                    "license_status": "present",
                    "redistribution_allowed": False,
                    "external_eval_only": True,
                    "admission_allowed": False,
                    "admission_reason": "external_eval_only",
                    "acquisition_date": None,
                    "sha256": None,
                },
            }
        }

        rows = build_license_audit_rows(manifest)
        by_source = {row["source"]: row for row in rows}

        self.assertIn("missing_acquisition_date", by_source["helpsteer2"]["risk_flags"])
        self.assertIn("missing_sha256", by_source["helpsteer2"]["risk_flags"])
        self.assertIn("external_eval_only", by_source["rewardbench"]["risk_flags"])
        self.assertIn("redistribution_restricted", by_source["rewardbench"]["risk_flags"])

    def test_validate_sci_gates_rejects_non_list_base_scores(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_sci_gates(
                base_scores={},
                repair_report={},
                validation_report={},
                ablation_report={},
                bias_report={},
                evidence_report={},
            )

    def test_count_unresolved_rows_accepts_list_or_int_reports(self):
        self.assertEqual(count_unresolved_rows({"unresolved_rows": ["a", "b"]}), 2)
        self.assertEqual(count_unresolved_rows({"unresolved_rows": 0}), 0)

    def test_validate_sci_gates_accepts_v2_dynamic_coverage(self):
        base_scores = [
            {
                "id": f"s{i}",
                "judge_backend": "m_prometheus",
                "parse_status": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            }
            for i in range(6946)
        ]
        report = validate_sci_gates(
            base_scores=base_scores,
            repair_report={
                "unresolved_rows": [],
                "coverage": {"required_pairwise_rows": 6946, "covered_pairwise_rows": 6946},
            },
            validation_report={
                "data_counts": {"factuality_train": 2500, "factuality_dev": 500, "factuality_test": 500},
                "validation_gate": {"passed": True},
                "backbone": {"base_judge": "prometheus_family_real_outputs"},
                "test_evaluation": {"factuality": {"metrics": {"macro_f1": 0.7162, "ece": 0.0153}}},
            },
            ablation_report={
                "local_prototype": False,
                "variants": [
                    {"name": name}
                    for name in [
                        "Full BEA-Judge",
                        "w/o Bias Module",
                        "w/o Evidence Module",
                        "w/o Calibration",
                        "w/o Base Judge Scores",
                        "w/o Tie Policy",
                        "w/o Review Threshold",
                    ]
                ],
            },
            bias_report={
                "prediction_coverage": {"coverage_ratio": 1.0},
                "validation_gates": {"risk_scores_in_range": True},
            },
            evidence_report={
                "summary": {"overall": {"profile_count": 3500}},
                "validation_gates": {"risk_scores_in_range": True},
            },
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["valid_pairwise_base_score_ids"], 6946)
        self.assertEqual(report["evidence_profile_count"], 3500)

    def test_build_ragtruth_result_rows_reports_error_taxonomy(self):
        payload = {
            "dev": {"factuality": [factuality_row("d1", "ragtruth", "unsupported", "supported")]},
            "test": {"factuality": [factuality_row("t1", "ragtruth", "supported", "unsupported")]},
        }

        rows = build_ragtruth_result_rows(payload)
        by_split = {row["split"]: row for row in rows}

        self.assertEqual(by_split["dev"]["unsupported_to_supported"], 1)
        self.assertEqual(by_split["test"]["supported_to_unsupported"], 1)

    def test_build_metric_ci_rows_outputs_bootstrap_intervals(self):
        payload = {
            "test": {
                "pairwise": [
                    pairwise_row("p1", "mt_bench", "Tie", "Tie"),
                    pairwise_row("p2", "mt_bench", "Tie", "A>B"),
                    pairwise_row("p3", "mt_bench", "A>B", "A>B"),
                ],
                "factuality": [factuality_row("f1", "ragtruth", "supported", "supported")],
            },
            "dev": {"pairwise": [], "factuality": []},
        }

        rows = build_metric_ci_rows(payload)
        pair_accuracy = [
            row for row in rows if row["head"] == "pairwise" and row["split"] == "test" and row["metric"] == "accuracy"
        ][0]

        self.assertEqual(pair_accuracy["point"], 0.6667)
        self.assertNotEqual(pair_accuracy["ci95_low"], "")
        self.assertNotEqual(pair_accuracy["ci95_high"], "")

    def test_mcnemar_pvalue_reports_exact_disagreement_test(self):
        self.assertEqual(mcnemar_exact_or_chi2_pvalue(0, 0), 1.0)
        self.assertLess(mcnemar_exact_or_chi2_pvalue(8, 0), 0.01)

    def test_generate_outputs_writes_expected_files(self):
        valid_base_scores = [
            {
                "id": f"s{i}",
                "judge_backend": "m_prometheus",
                "parse_status": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            }
            for i in range(2646)
        ]
        validation_report = {
            "data_counts": {
                "pairwise_dev": 2,
                "pairwise_test": 2,
                "factuality_dev": 1,
                "factuality_test": 1,
            },
            "heads": {
                "pairwise": {"calibrated_dev_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0}},
                "factuality": {"calibrated_dev_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": None}},
            },
            "test_evaluation": {
                "pairwise": {"metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0}},
                "factuality": {"metrics": {"accuracy": 0.9, "macro_f1": 0.9, "ece": 0.01, "brier": 0.1, "tie_recall": None}},
            },
            "validation_gate": {"passed": True},
            "backbone": {"base_judge": "prometheus_family_real_outputs"},
        }
        calibrated = {
            "dev": {"pairwise": [pairwise_row("d1", "mt_bench", "Tie", "Tie")], "factuality": [factuality_row("fd1", "ares_nq", "supported", "supported")]},
            "test": {"pairwise": [pairwise_row("t1", "mt_bench", "Tie", "Tie")], "factuality": [factuality_row("ft1", "ares_nq", "supported", "supported")]},
        }
        ablation_report = {
            "local_prototype": False,
            "variants": [
                {
                    "name": name,
                    "pairwise": {
                        "dev_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0, "gold_distribution": {"Tie": 1}, "pred_distribution": {"Tie": 1}},
                        "test_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0, "gold_distribution": {"Tie": 1}, "pred_distribution": {"Tie": 1}},
                    },
                    "factuality": {
                        "dev_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": None},
                        "test_metrics": {"accuracy": 0.9, "macro_f1": 0.9, "ece": 0.01, "brier": 0.1, "tie_recall": None},
                    },
                }
                for name in [
                    "Full BEA-Judge",
                    "w/o Bias Module",
                    "w/o Evidence Module",
                    "w/o Calibration",
                    "w/o Base Judge Scores",
                    "w/o Tie Policy",
                    "w/o Review Threshold",
                ]
            ],
            "control_baselines": [
                {
                    "name": "Raw M-Prometheus-3B only",
                    "pairwise": {
                        "dev_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0, "gold_distribution": {"Tie": 1}, "pred_distribution": {"Tie": 1}},
                        "test_metrics": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.0, "brier": 0.0, "tie_recall": 1.0, "gold_distribution": {"Tie": 1}, "pred_distribution": {"Tie": 1}},
                        "dev_rows": [pairwise_row("d1", "mt_bench", "Tie", "Tie")],
                        "test_rows": [pairwise_row("t1", "mt_bench", "Tie", "Tie")],
                    },
                }
            ],
            "bias_utility": [
                {
                    "setting": "bias_risk_only_review",
                    "head": "pairwise",
                    "split": "test",
                    "n": 1,
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "ece": 0.0,
                    "review_rate": 0.0,
                    "review_capture_rate": "",
                }
            ],
        }
        repair_report = {
            "replaced_rows": 34,
            "unresolved_rows": [],
            "coverage": {"required_pairwise_rows": 2646, "covered_pairwise_rows": 2646},
        }
        bias_report = {
            "prediction_coverage": {"coverage_ratio": 1.0},
            "validation_gates": {"risk_scores_in_range": True},
        }
        bias_profiles = {
            "profiles": [
                {
                    "id": "t1",
                    "dataset": "mt_bench",
                    "task_type": "open_qa",
                    "split": "test",
                    "metadata": {"bias_type": "none"},
                    "prediction": {"predicted_label": "Tie", "gold_label": "Tie", "confidence": 0.8},
                    "bias": {"overall_bias_risk": 0.1, "review_required": False},
                }
            ]
        }
        evidence_report = {
            "summary": {"overall": {"profile_count": 1000}},
            "validation_gates": {"risk_scores_in_range": True},
        }
        evidence_profiles = {
            "profiles": [
                {
                    "id": "ft1",
                    "evidence": {"evidence_risk": 0.1, "reasons": []},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = {
                "base": tmp / "base_scores.repaired.json",
                "repair": tmp / "repair.json",
                "validation": tmp / "validation.json",
                "calibrated": tmp / "calibrated.json",
                "ablation": tmp / "ablation.json",
                "bias": tmp / "bias.json",
                "bias_profiles": tmp / "bias_profiles.json",
                "evidence": tmp / "evidence.json",
                "evidence_profiles": tmp / "evidence_profiles.json",
                "swap": tmp / "swap_probe_report.json",
                "expansion": tmp / "expansion_v2_report.json",
                "manifest_v2": tmp / "data_manifest_v2.json",
                "calibration": tmp / "calibration_per_method.json",
                "out": tmp / "out",
            }
            swap_report = {
                "dataset_summary": [
                    {
                        "dataset": "overall",
                        "selected_n": 1,
                        "swap_available_n": 1,
                        "swap_parse_success_rate": 1.0,
                        "swap_consistency_rate": 1.0,
                        "swap_inconsistency_rate": 0.0,
                        "calibrated_error_rate": 0.0,
                        "error_rate_when_inconsistent": "",
                        "avg_swap_margin_delta": 0.0,
                        "tie_case_rate": 1.0,
                    }
                ]
            }
            expansion_report = {
                "statistics": {
                    "total_samples": 3401,
                    "by_task_type": {"open_qa": 1201},
                    "by_dataset": {"helpsteer2": 1},
                    "by_split": {"train": 1},
                    "by_language": {"en": 1},
                    "human_label_distribution": {"A>B": 1},
                },
                "source_reports": [{"source": "helpsteer2", "accepted_records": 1}],
            }
            manifest_v2 = {
                "sources": {
                    "helpsteer2": {
                        "license": "CC-BY-4.0",
                        "redistribution_allowed": True,
                        "admission_allowed": True,
                        "admission_reason": "ok",
                        "acquisition_date": "2026-05-18T00:00:00+00:00",
                        "sha256": "abc",
                    }
                }
            }
            payloads = {
                paths["base"]: valid_base_scores,
                paths["repair"]: repair_report,
                paths["validation"]: validation_report,
                paths["calibrated"]: calibrated,
                paths["ablation"]: ablation_report,
                paths["bias"]: bias_report,
                paths["bias_profiles"]: bias_profiles,
                paths["evidence"]: evidence_report,
                paths["evidence_profiles"]: evidence_profiles,
                paths["swap"]: swap_report,
                paths["expansion"]: expansion_report,
                paths["manifest_v2"]: manifest_v2,
                paths["calibration"]: {
                    "results": {
                        "temperature": {
                            "metrics_dev": {"accuracy": 1.0, "ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0},
                            "metrics_test": {"accuracy": 1.0, "ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0},
                            "extras": {},
                        }
                    }
                },
            }
            for path, payload in payloads.items():
                path.write_text(json.dumps(payload), encoding="utf-8")

            index = generate_outputs(
                base_scores_path=paths["base"],
                repair_report_path=paths["repair"],
                validation_report_path=paths["validation"],
                calibrated_results_path=paths["calibrated"],
                ablation_report_path=paths["ablation"],
                bias_report_path=paths["bias"],
                bias_profiles_path=paths["bias_profiles"],
                evidence_report_path=paths["evidence"],
                evidence_profiles_path=paths["evidence_profiles"],
                output_dir=paths["out"],
                swap_report_path=paths["swap"],
                expansion_report_path=paths["expansion"],
                manifest_v2_path=paths["manifest_v2"],
                calibration_comparison_path=paths["calibration"],
            )

            self.assertTrue((paths["out"] / "main_results_table.csv").exists())
            self.assertTrue((paths["out"] / "baseline_comparison_table.csv").exists())
            self.assertTrue((paths["out"] / "metric_confidence_interval_table.csv").exists())
            self.assertTrue((paths["out"] / "ablation_table.md").exists())
            self.assertTrue((paths["out"] / "ablation_significance_table.csv").exists())
            self.assertTrue((paths["out"] / "bias_risk_utility_table.csv").exists())
            self.assertTrue((paths["out"] / "risk_coverage_table.csv").exists())
            self.assertTrue((paths["out"] / "calibration_methods_table.csv").exists())
            self.assertTrue((paths["out"] / "tie_recall_table.csv").exists())
            self.assertTrue((paths["out"] / "per_dataset_table.md").exists())
            self.assertTrue((paths["out"] / "base_diagnostics_table.md").exists())
            self.assertTrue((paths["out"] / "bias_subgroup_table.csv").exists())
            self.assertTrue((paths["out"] / "evidence_subtype_table.csv").exists())
            self.assertTrue((paths["out"] / "swap_consistency_table.csv").exists())
            self.assertTrue((paths["out"] / "source_provenance_table.csv").exists())
            self.assertTrue((paths["out"] / "license_audit_table.csv").exists())
            self.assertTrue((paths["out"] / "v2_distribution_table.csv").exists())
            self.assertTrue((paths["out"] / "data_scaling_table.csv").exists())
            self.assertTrue((paths["out"] / "method_summary.md").exists())
            self.assertTrue((paths["out"] / "data_availability_statement.md").exists())
            self.assertTrue(index["gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
