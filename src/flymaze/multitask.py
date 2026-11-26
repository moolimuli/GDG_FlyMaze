"""Behavior-warm-started, PPO-fine-tuned multi-map policy for extended MVP validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from flymaze.agents import StatelessRuleAgent
from flymaze.config import MultitaskTrainingConfig, PPOHyperparameters
from flymaze.env import FlyMazeEnv
from flymaze.evaluation import evaluate_agent
from flymaze.ppo import SB3AgentAdapter, SymmetricPPOActionWrapper
from flymaze.relative_odor_agent import RelativeOdorRuleAgent

MULTITASK_TRAIN_MAP_IDS = (
    "train_l_left",
    "train_l_right",
    "validation_s_turn_complex",
)


RULE_EXPERTS: dict[str, Callable[[], StatelessRuleAgent | RelativeOdorRuleAgent]] = {
    "original": StatelessRuleAgent,
    "relative": RelativeOdorRuleAgent,
}


def make_rule_expert(name: str) -> StatelessRuleAgent | RelativeOdorRuleAgent:
    """Build the demonstration expert; "original" keeps the released behavior."""

    try:
        return RULE_EXPERTS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(RULE_EXPERTS))
        raise ValueError(f"unknown rule expert: {name}; choose from {choices}") from exc


@dataclass(frozen=True, slots=True)
class DemonstrationSummary:
    map_id: str
    seed: int
    success: bool
    steps: int


def public_to_symmetric_action(action: NDArray[np.float32]) -> NDArray[np.float32]:
    values = np.asarray(action, dtype=np.float32).copy()
    values[0] = 2.0 * values[0] - 1.0
    return values


def collect_rule_demonstrations(
    map_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    samples_per_demonstration: int,
    expert: str = "original",
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[DemonstrationSummary]]:
    observations: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    summaries: list[DemonstrationSummary] = []
    for map_id in map_ids:
        env = FlyMazeEnv(map_id)
        try:
            for seed in seeds:
                episode_observations: list[NDArray[np.float32]] = []
                episode_actions: list[NDArray[np.float32]] = []
                agent = make_rule_expert(expert)
                observation, _ = env.reset(seed=seed)
                terminated = False
                truncated = False
                steps = 0
                info: dict[str, object] = {"success": False}
                while not (terminated or truncated):
                    action = agent.predict(observation)
                    episode_observations.append(observation.copy())
                    episode_actions.append(public_to_symmetric_action(action))
                    observation, _, terminated, truncated, info = env.step(action)
                    steps += 1
                success = bool(info.get("success", False))
                summaries.append(DemonstrationSummary(map_id, seed, success, steps))
                if not success:
                    raise RuntimeError(f"Rule-based demonstration failed: {map_id} seed={seed}")
                balanced_indices = np.arange(samples_per_demonstration) % len(episode_observations)
                observations.extend(episode_observations[index] for index in balanced_indices)
                actions.extend(episode_actions[index] for index in balanced_indices)
        finally:
            env.close()
    return np.asarray(observations), np.asarray(actions), summaries


def _make_map_training_env(map_id: str) -> SymmetricPPOActionWrapper:
    return SymmetricPPOActionWrapper(FlyMazeEnv(map_id))


def _fit_behavior_policy(
    model: PPO,
    observations: NDArray[np.float32],
    actions: NDArray[np.float32],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    model.policy.set_training_mode(True)
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(epochs):
        indices = generator.permutation(len(observations))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            distribution = model.policy.get_distribution(observation_tensor[batch_indices])
            base_distribution = distribution.distribution
            if not isinstance(base_distribution, torch.distributions.Normal):
                raise TypeError("PPO behavior warm start requires a diagonal Normal policy")
            predicted_mean = base_distribution.mean
            loss = torch.nn.functional.mse_loss(predicted_mean, action_tensor[batch_indices])
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return losses


def train_multitask_ppo(
    *,
    output_directory: Path,
    hyperparameters: PPOHyperparameters,
    training: MultitaskTrainingConfig,
    expert: str = "original",
) -> dict[str, object]:
    """Warm-start an SB3 PPO actor from demonstrations, then continue on-policy PPO updates."""

    rollout_size = training.n_envs * hyperparameters.n_steps
    if rollout_size % hyperparameters.batch_size != 0:
        raise ValueError("batch_size must divide n_envs * n_steps")
    output_directory.mkdir(parents=True, exist_ok=True)
    observations, actions, demo_summaries = collect_rule_demonstrations(
        MULTITASK_TRAIN_MAP_IDS,
        training.demonstration_seeds,
        training.samples_per_demonstration,
        expert,
    )
    base_env = SymmetricPPOActionWrapper(FlyMazeEnv("train_l_left"))
    model = PPO(
        "MlpPolicy",
        base_env,
        learning_rate=hyperparameters.learning_rate,
        n_steps=hyperparameters.n_steps,
        batch_size=hyperparameters.batch_size,
        gamma=hyperparameters.gamma,
        gae_lambda=hyperparameters.gae_lambda,
        ent_coef=hyperparameters.entropy_coefficient,
        seed=training.seed,
        verbose=1,
        tensorboard_log=str(output_directory / "tensorboard"),
        policy_kwargs={"net_arch": list(hyperparameters.network)},
        device=hyperparameters.device,
    )
    try:
        batch_losses = _fit_behavior_policy(
            model,
            observations,
            actions,
            epochs=training.behavior_epochs,
            batch_size=hyperparameters.batch_size,
            seed=training.seed,
        )
        behavior_path = output_directory / "ppo_behavior_warm_start"
        model.save(behavior_path)
    finally:
        base_env.close()

    env_functions: list[Callable[[], Env]] = [
        partial(
            _make_map_training_env, MULTITASK_TRAIN_MAP_IDS[index % len(MULTITASK_TRAIN_MAP_IDS)]
        )
        for index in range(training.n_envs)
    ]
    vec_env = SubprocVecEnv(env_functions, start_method="forkserver")
    vec_env.seed(training.seed)
    try:
        model = PPO.load(behavior_path, env=vec_env, device=hyperparameters.device)
        model.learn(
            total_timesteps=training.ppo_timesteps,
            reset_num_timesteps=True,
            tb_log_name="PPO_multitask_finetune",
            progress_bar=False,
        )
        post_ppo_losses = _fit_behavior_policy(
            model,
            observations,
            actions,
            epochs=training.post_ppo_behavior_epochs,
            batch_size=hyperparameters.batch_size,
            seed=training.seed + 1,
        )
        final_path = output_directory / "ppo_multitask_final"
        model.save(final_path)
    finally:
        vec_env.close()

    actual_transitions = model.num_timesteps
    reloaded = PPO.load(final_path, device=hyperparameters.device)
    adapter = SB3AgentAdapter(reloaded)
    evaluations = {
        map_id: evaluate_agent(map_id, adapter, training.evaluation_seeds)
        for map_id in (*MULTITASK_TRAIN_MAP_IDS, "test_straight_unknown")
    }
    return {
        "hyperparameters": asdict(hyperparameters),
        "training": asdict(training),
        "expert": expert,
        "demonstration_count": len(observations),
        "demonstrations": [asdict(summary) for summary in demo_summaries],
        "behavior_epochs": training.behavior_epochs,
        "behavior_loss_mean": float(np.mean(batch_losses)),
        "behavior_loss_final": batch_losses[-1],
        "post_ppo_behavior_epochs": training.post_ppo_behavior_epochs,
        "post_ppo_behavior_loss_final": post_ppo_losses[-1],
        "behavior_checkpoint": str(behavior_path.with_suffix(".zip")),
        "requested_ppo_timesteps": training.ppo_timesteps,
        "actual_ppo_transitions": actual_transitions,
        "model_path": str(final_path.with_suffix(".zip")),
        "evaluations": evaluations,
    }
