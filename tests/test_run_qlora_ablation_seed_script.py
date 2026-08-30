import subprocess
import unittest
from pathlib import Path

from shell_test_utils import find_bash


ROOT = Path(__file__).resolve().parents[1]


class RunQloraAblationSeedScriptTest(unittest.TestCase):
    def test_shell_script_has_valid_syntax(self) -> None:
        bash = find_bash()
        if bash is None:
            self.skipTest("a Bash interpreter is required for shell syntax validation")
        result = subprocess.run(
            [bash, "-n", str(ROOT / "scripts" / "run_qlora_ablation_seed.sh")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_shell_script_supports_bounded_score_smoke(self) -> None:
        content = (ROOT / "scripts" / "run_qlora_ablation_seed.sh").read_text(encoding="utf-8")

        self.assertIn('MAX_SCORE_SAMPLES="${MAX_SCORE_SAMPLES:-}"', content)
        self.assertIn('MAX_FUSION_SAMPLES="${MAX_FUSION_SAMPLES:-${MAX_SCORE_SAMPLES:-}}"', content)
        self.assertIn('SCORE_ARGS=()', content)
        self.assertIn('FUSION_ARGS=()', content)
        self.assertIn('--limit "$MAX_SCORE_SAMPLES"', content)
        self.assertIn('--sample-limit "$MAX_FUSION_SAMPLES"', content)


if __name__ == "__main__":
    unittest.main()
