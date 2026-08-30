import subprocess
import unittest
from pathlib import Path

from shell_test_utils import find_bash


ROOT = Path(__file__).resolve().parents[1]


class RunQloraAblationGridScriptTest(unittest.TestCase):
    def test_shell_script_has_valid_syntax(self) -> None:
        bash = find_bash()
        if bash is None:
            self.skipTest("a Bash interpreter is required for shell syntax validation")
        result = subprocess.run(
            [bash, "-n", str(ROOT / "scripts" / "run_qlora_ablation_grid.sh")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_shell_script_contains_epoch_and_sft_protocols(self) -> None:
        content = (ROOT / "scripts" / "run_qlora_ablation_grid.sh").read_text(encoding="utf-8")

        self.assertIn("epoch0p5_1024", content)
        self.assertIn("epoch2_1024", content)
        self.assertIn("sft25_epoch1_1024", content)
        self.assertIn("sft50_epoch1_1024", content)
        self.assertIn("sft100_epoch1_1024", content)
        self.assertIn("alias_from_setting", content)


if __name__ == "__main__":
    unittest.main()
