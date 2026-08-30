import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_external_3b_baselines import (  # noqa: E402
    build_glider_prompt,
    build_prometheus2_prompt,
    build_pairwise_prompt,
    build_reward_text,
    grm_label_from_scores,
    metrics_for_predictions,
    parse_args,
    select_grm_margin,
    softmax,
)


class RunExternal3BBaselinesTest(unittest.TestCase):
    def test_softmax_normalizes_probabilities(self) -> None:
        probs = softmax([1.0, 2.0, 3.0])

        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[2], probs[1])
        self.assertGreater(probs[1], probs[0])

    def test_grm_label_uses_dev_selected_margin_rule(self) -> None:
        pred, probs = grm_label_from_scores(0.7, 0.2, margin=0.1)
        self.assertEqual(pred, "A>B")
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

        pred, probs = grm_label_from_scores(0.7, 0.65, margin=0.1)
        self.assertEqual(pred, "Tie")
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

        pred, _ = grm_label_from_scores(0.1, 0.4, margin=0.1)
        self.assertEqual(pred, "B>A")

    def test_select_grm_margin_prefers_macro_f1_on_dev_rows(self) -> None:
        rows = [
            {"gold_label": "A>B", "score_a": 0.9, "score_b": 0.1},
            {"gold_label": "B>A", "score_a": 0.1, "score_b": 0.9},
            {"gold_label": "Tie", "score_a": 0.6, "score_b": 0.59},
            {"gold_label": "Tie", "score_a": 0.51, "score_b": 0.5},
        ]

        margin, candidates = select_grm_margin(rows)

        self.assertGreaterEqual(margin, 0.0)
        self.assertTrue(candidates)
        best = max(candidates, key=lambda row: row["metrics"]["macro_f1"])
        self.assertEqual(margin, best["margin"])

    def test_metrics_include_tie_recall_and_parse_failure(self) -> None:
        rows = [
            {"gold_label": "A>B", "pred_label": "A>B", "confidence": 0.8},
            {"gold_label": "B>A", "pred_label": "A>B", "confidence": 0.7},
            {"gold_label": "Tie", "pred_label": "Tie", "confidence": 0.6},
            {"gold_label": "Tie", "pred_label": None, "confidence": 0.0},
        ]

        metrics = metrics_for_predictions(rows)

        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["valid_n"], 3)
        self.assertEqual(metrics["parse_failure_rate"], 0.25)
        self.assertEqual(metrics["tie_recall"], 1.0)
        self.assertTrue(0.0 <= metrics["ece"] <= 1.0)

    def test_qwen_prompt_contains_all_required_labels(self) -> None:
        prompt = build_pairwise_prompt(
            {
                "prompt": "Explain gravity.",
                "context": "Physics question.",
                "reference": "Masses attract.",
                "answer_a": "Gravity attracts masses.",
                "answer_b": "Gravity is unrelated to mass.",
            }
        )

        self.assertIn("[RESULT] A", prompt)
        self.assertIn("[RESULT] B", prompt)
        self.assertIn("[RESULT] Tie", prompt)
        self.assertIn("Response A:", prompt)
        self.assertIn("Response B:", prompt)

    def test_reward_text_uses_chat_roles(self) -> None:
        text = build_reward_text(
            {
                "prompt": "Explain gravity.",
                "context": "Physics question.",
                "reference": "Masses attract.",
                "answer_a": "Gravity attracts masses.",
            },
            "answer_a",
        )

        self.assertIn("User:", text)
        self.assertIn("Assistant:", text)
        self.assertIn("Gravity attracts masses.", text)

    def test_prometheus2_prompt_contains_pairwise_ranking_fields(self) -> None:
        prompt = build_prometheus2_prompt(
            {
                "prompt": "Explain gravity.",
                "context": "Physics question.",
                "reference": "Masses attract.",
                "answer_a": "Gravity attracts masses.",
                "answer_b": "Gravity is unrelated to mass.",
            }
        )

        self.assertIn("Prometheus 2", prompt)
        self.assertIn("Return exactly one letter: A if Response A is better, or B if Response B is better.", prompt)
        self.assertIn("Reference answer:", prompt)
        self.assertIn("Response A:", prompt)
        self.assertIn("Response B:", prompt)

    def test_argparse_accepts_prometheus2_pairwise(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_external_3b_baselines.py",
                "--model-kind",
                "prometheus2_pairwise",
                "--model-path",
                "models/external_baselines/prometheus-7b-v2.0",
                "--model-name",
                "prometheus-eval/prometheus-7b-v2.0",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.model_kind, "prometheus2_pairwise")
        self.assertEqual(args.model_name, "prometheus-eval/prometheus-7b-v2.0")

    def test_glider_prompt_contains_rubric_fields(self) -> None:
        prompt = build_glider_prompt(
            {
                "prompt": "Explain gravity.",
                "context": "Physics question.",
                "reference": "Masses attract.",
                "answer_a": "Gravity attracts masses.",
                "answer_b": "Gravity is unrelated to mass.",
            }
        )

        self.assertIn("Data:", prompt)
        self.assertIn("Pass criteria:", prompt)
        self.assertIn("Rubric:", prompt)
        self.assertIn("Response A:", prompt)
        self.assertIn("Response B:", prompt)
        self.assertIn("Score 1: Response A is better than Response B.", prompt)
        self.assertIn("Score 2: Response A and Response B are approximately tied.", prompt)
        self.assertIn("Score 3: Response B is better than Response A.", prompt)

    def test_argparse_accepts_glider_evaluator(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_external_3b_baselines.py",
                "--model-kind",
                "glider_evaluator",
                "--model-path",
                "models/external_baselines/glider",
                "--model-name",
                "PatronusAI/glider",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.model_kind, "glider_evaluator")
        self.assertEqual(args.model_name, "PatronusAI/glider")


if __name__ == "__main__":
    unittest.main()
