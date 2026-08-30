"""Render the next QLoRA-BEA-Judge experiment matrix as a Markdown plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
DEFAULT_CONFIG = ROOT / "configs" / "qlora_next_experiments.json"
DEFAULT_OUTPUT = (
    ROOT
    / "datasets"
    / "model_outputs"
    / "qlora_3seed_epoch1_1024_summary"
    / "next_experiment_plan.md"
)


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def comma_join(values: Sequence[Any]) -> str:
    return ", ".join(str(value) for value in values)


def build_markdown(config: Mapping[str, Any]) -> str:
    stable = config["stable_protocol"]
    policy = config["reporting_policy"]
    lines = [
        "# QLoRA-BEA-Judge Next Experiment Plan",
        "",
        f"Version: {config['version']}",
        "",
        "## Objective",
        "",
        str(config["objective"]),
        "",
        "## Stable Protocol",
        "",
        f"- GPU profile: {stable['gpu_profile']}",
        f"- Config: `{stable['config']}`",
        f"- Seeds: {comma_join(stable['seeds'])}",
        f"- Max sequence length: {stable['max_seq_length']}",
        f"- Runner: `{stable['runner']}`",
        f"- Submission summary: `{stable['submission_summary']}`",
        f"- Validation command: `{stable['validation_command']}`",
        "",
        "## Reporting Policy",
        "",
        f"- Primary claim: {policy['primary_claim']}",
        f"- Secondary claim: {policy['secondary_claim']}",
        f"- Constraint: {policy['constraint']}",
        "",
        "## Experiment Matrix",
        "",
    ]
    for group in config["experiment_groups"]:
        lines.extend(
            [
                f"### {group['priority']} {group['id']}",
                "",
                str(group["purpose"]),
                "",
                "| experiment | table | metrics | gate keys |",
                "| --- | --- | --- | --- |",
            ]
        )
        for experiment in group["experiments"]:
            gate_keys = comma_join(experiment["gate"].keys())
            lines.append(
                f"| {experiment['id']} | {experiment.get('table', '')} | "
                f"{comma_join(experiment['metrics'])} | {gate_keys} |"
            )
        lines.extend(["", "Commands:", ""])
        for experiment in group["experiments"]:
            lines.extend(
                [
                    f"- `{experiment['id']}`",
                    "",
                    "```bash",
                    str(experiment["command"]),
                    "```",
                    "",
                ]
            )
            if experiment.get("postprocess_command"):
                lines.extend(
                    [
                        "Postprocess:",
                        "",
                        "```bash",
                        str(experiment["postprocess_command"]),
                        "```",
                        "",
                    ]
                )
            if experiment.get("smoke_command"):
                lines.extend(
                    [
                        "Smoke:",
                        "",
                        "```bash",
                        str(experiment["smoke_command"]),
                        "```",
                        "",
                    ]
                )
        artifact_lines = []
        for experiment in group["experiments"]:
            for artifact in experiment.get("existing_artifacts", []):
                artifact_lines.append(f"- `{artifact}`")
        if artifact_lines:
            lines.extend(["Existing artifacts:", "", *artifact_lines, ""])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown plan from the QLoRA experiment matrix.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(resolve_root_path(args.config))
    output = resolve_root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(config), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
