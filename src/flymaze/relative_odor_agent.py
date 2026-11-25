"""Memoryless expert that steers by the relative left/right odor difference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RelativeOdorRuleAgent:
    """Same rules as StatelessRuleAgent, but odor is compared as a ratio.

    The original expert uses the absolute difference ``odor_left - odor_right``.
    Far from the food that difference is tiny, so the expert cannot tell the
    direction. Here the difference is divided by the total, which stays usable
    along the whole route. Only the standard seven observations are used.
    """

    def __init__(
        self,
        relative_turn_gain: float = 20.0,
        obstacle_relative_threshold: float = 0.005,
        epsilon: float = 1e-12,
    ) -> None:
        self.relative_turn_gain = relative_turn_gain
        self.obstacle_relative_threshold = obstacle_relative_threshold
        self.epsilon = epsilon

    def relative_odor_difference(self, odor_left: float, odor_right: float) -> float:
        # Antennae are excluded from wall collision, so one can poke into a wall
        # and read exactly 0. A ratio would turn that into a full-strength (+/-1)
        # signal, so a reading of 0 on either side counts as "no odor information".
        if odor_left <= 0.0 or odor_right <= 0.0:
            return 0.0
        return (odor_left - odor_right) / (odor_left + odor_right + self.epsilon)

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError("RelativeOdorRuleAgent requires the seven-dimensional observation")
        front, left_obstacle, right_obstacle, odor_left, odor_right, _, _ = values
        relative = self.relative_odor_difference(odor_left, odor_right)
        if min(front, left_obstacle, right_obstacle) < 1.0:
            if abs(relative) >= self.obstacle_relative_threshold:
                turn = 1.0 if relative > 0.0 else -1.0
            else:
                turn = -1.0 if left_obstacle < right_obstacle else 1.0
            return np.asarray([0.5, turn], dtype=np.float32)
        turn = float(np.clip(self.relative_turn_gain * relative, -1.0, 1.0))
        return np.asarray([1.0 - 0.5 * abs(turn), turn], dtype=np.float32)
