from flymaze.cli import build_parser


def test_cli_exposes_only_supported_ppo_workflows() -> None:
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    assert set(subparsers_action.choices) == {
        "doctor",
        "train-ppo",
        "train-multitask-ppo",
        "evaluate-ppo",
        "visualize-ppo",
    }


def test_cli_override_values_are_optional_until_config_resolution() -> None:
    args = build_parser().parse_args(
        [
            "train-ppo",
            "--config",
            "configs/ppo.toml",
            "--output-directory",
            "artifacts/test",
            "--seed",
            "9",
        ]
    )
    assert args.seed == 9
    assert args.n_envs is None


def test_multitask_expert_defaults_to_the_released_original() -> None:
    base = ["train-multitask-ppo", "--config", "configs/ppo.toml", "--output-directory", "x"]
    assert build_parser().parse_args(base).expert == "original"
    assert build_parser().parse_args([*base, "--expert", "relative"]).expert == "relative"
