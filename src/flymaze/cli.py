"""Focused command-line interface for PPO training, evaluation, and review."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from flymaze.config import FlyMazeConfig, PPOHyperparameters
from flymaze.doctor import environment_report
from flymaze.evaluation import evaluate_agent
from flymaze.multitask import train_multitask_ppo
from flymaze.ppo import SB3AgentAdapter, train_ppo
from flymaze.visualization import run_visual_review


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_report(report: dict[str, object], output: Path | None = None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _overridden_hyperparameters(
    configured: PPOHyperparameters, args: argparse.Namespace
) -> PPOHyperparameters:
    updates: dict[str, Any] = {}
    for field, argument in (
        ("learning_rate", "learning_rate"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("entropy_coefficient", "entropy_coefficient"),
        ("n_steps", "n_steps"),
        ("batch_size", "batch_size"),
        ("network", "network"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            updates[field] = tuple(value) if field == "network" else value
    return replace(configured, **updates)


def _doctor(_: argparse.Namespace) -> int:
    _render_report(environment_report())
    return 0


def _train_ppo(args: argparse.Namespace) -> int:
    configured = FlyMazeConfig.from_toml(args.config)
    hyperparameters = _overridden_hyperparameters(configured.ppo, args)
    training = replace(
        configured.scratch,
        **{
            name: value
            for name, value in {
                "total_timesteps": args.total_timesteps,
                "seed": args.seed,
                "n_envs": args.n_envs,
                "evaluation_seeds": (
                    None if args.evaluation_seeds is None else tuple(args.evaluation_seeds)
                ),
            }.items()
            if value is not None
        },
    )
    output_directory = Path(args.output_directory)
    report = train_ppo(hyperparameters, training, output_directory)
    model_path = Path(str(report["model_path"]))
    report["model_sha256"] = _sha256(model_path)
    _render_report(report, output_directory / "training_report.json")
    return 0


def _train_multitask_ppo(args: argparse.Namespace) -> int:
    configured = FlyMazeConfig.from_toml(args.config)
    hyperparameters = _overridden_hyperparameters(configured.ppo, args)
    training = replace(
        configured.multitask,
        **{
            name: value
            for name, value in {
                "seed": args.seed,
                "demonstration_seeds": (
                    None if args.demonstration_seeds is None else tuple(args.demonstration_seeds)
                ),
                "behavior_epochs": args.behavior_epochs,
                "post_ppo_behavior_epochs": args.post_ppo_behavior_epochs,
                "ppo_timesteps": args.ppo_timesteps,
                "n_envs": args.n_envs,
                "samples_per_demonstration": args.samples_per_demonstration,
                "evaluation_seeds": (
                    None if args.evaluation_seeds is None else tuple(args.evaluation_seeds)
                ),
            }.items()
            if value is not None
        },
    )
    output_directory = Path(args.output_directory)
    report = train_multitask_ppo(
        output_directory=output_directory,
        hyperparameters=hyperparameters,
        training=training,
        expert=args.expert,
    )
    model_path = Path(str(report["model_path"]))
    report["model_sha256"] = _sha256(model_path)
    _render_report(report, output_directory / "training_report.json")
    return 0


def _evaluate_ppo(args: argparse.Namespace) -> int:
    model_path = Path(args.model_path)
    model = PPO.load(model_path, device="cpu")
    report = evaluate_agent(args.map_id, SB3AgentAdapter(model), tuple(args.seeds))
    report["model_path"] = str(model_path)
    report["model_sha256"] = _sha256(model_path)
    _render_report(report, None if args.output is None else Path(args.output))
    return 0


def _visualize_ppo(args: argparse.Namespace) -> int:
    report = run_visual_review(
        map_id=args.map_id,
        seed=args.seed,
        model_path=Path(args.model_path),
        playback_speed=args.playback_speed,
        hold_seconds=args.hold_seconds,
        persist=args.persist,
        show_contacts=args.show_contacts,
        show_odor=args.show_odor,
        odor_stride=args.odor_stride,
        odor_alpha=args.odor_alpha,
    )
    _render_report(report)
    return 0


def _add_shared_ppo_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--network", type=int, nargs="+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flymaze")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="print reproducibility metadata")
    doctor.set_defaults(handler=_doctor)

    scratch = subparsers.add_parser("train-ppo", help="train PPO from random initialization")
    scratch.add_argument("--config", required=True)
    scratch.add_argument("--output-directory", required=True)
    scratch.add_argument("--total-timesteps", type=int)
    scratch.add_argument("--seed", type=int)
    scratch.add_argument("--n-envs", type=int)
    scratch.add_argument("--evaluation-seeds", type=int, nargs="+")
    _add_shared_ppo_overrides(scratch)
    scratch.set_defaults(handler=_train_ppo)

    multitask = subparsers.add_parser(
        "train-multitask-ppo",
        help="warm-start from memoryless demonstrations and fine-tune with PPO",
    )
    multitask.add_argument("--config", required=True)
    multitask.add_argument("--output-directory", required=True)
    multitask.add_argument("--seed", type=int)
    multitask.add_argument("--demonstration-seeds", type=int, nargs="+")
    multitask.add_argument("--behavior-epochs", type=int)
    multitask.add_argument("--post-ppo-behavior-epochs", type=int)
    multitask.add_argument("--ppo-timesteps", type=int)
    multitask.add_argument("--n-envs", type=int)
    multitask.add_argument("--samples-per-demonstration", type=int)
    multitask.add_argument("--evaluation-seeds", type=int, nargs="+")
    multitask.add_argument(
        "--expert",
        choices=["original", "relative"],
        default="original",
        help="rule expert for demonstrations; original keeps the released behavior",
    )
    _add_shared_ppo_overrides(multitask)
    multitask.set_defaults(handler=_train_multitask_ppo)

    evaluate = subparsers.add_parser("evaluate-ppo", help="evaluate a saved PPO checkpoint")
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--map-id", required=True)
    evaluate.add_argument("--seeds", type=int, nargs="+", required=True)
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_evaluate_ppo)

    visualize = subparsers.add_parser("visualize-ppo", help="open the native MuJoCo PPO viewer")
    visualize.add_argument("--model-path", required=True)
    visualize.add_argument("--map-id", default="train_l_left")
    visualize.add_argument("--seed", type=int, default=0)
    visualize.add_argument("--playback-speed", type=float, default=1.0)
    visualize.add_argument("--hold-seconds", type=float, default=5.0)
    visualize.add_argument("--persist", action="store_true")
    visualize.add_argument("--show-contacts", action="store_true")
    visualize.add_argument("--show-odor", action="store_true")
    visualize.add_argument("--odor-stride", type=int, default=4)
    visualize.add_argument("--odor-alpha", type=float, default=0.35)
    visualize.set_defaults(handler=_visualize_ppo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))
