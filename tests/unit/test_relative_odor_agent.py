import numpy as np
import pytest

from flymaze.agents import StatelessRuleAgent
from flymaze.relative_odor_agent import RelativeOdorRuleAgent


def _observation(odor_left: float, odor_right: float, obstacles: float = 1.0) -> np.ndarray:
    return np.asarray(
        [obstacles, obstacles, obstacles, odor_left, odor_right, 0.0, 0.0], dtype=np.float32
    )


def test_faint_odor_still_steers_toward_the_stronger_side() -> None:
    # Left is ~10% stronger, but far from the food (absolute values ~4e-5).
    faint = _observation(4.2e-5, 3.8e-5)
    relative_turn = RelativeOdorRuleAgent().predict(faint)[1]
    original_turn = StatelessRuleAgent().predict(faint)[1]
    assert relative_turn > 0.9
    assert abs(original_turn) < 0.01


def test_scale_invariance_far_and_near_give_the_same_action() -> None:
    agent = RelativeOdorRuleAgent()
    far = agent.predict(_observation(4.1e-5, 3.9e-5))
    near = agent.predict(_observation(0.41, 0.39))
    np.testing.assert_allclose(far, near, rtol=1e-4)


def test_obstacle_mode_uses_relative_odor_direction() -> None:
    agent = RelativeOdorRuleAgent()
    blocked = np.asarray([0.5, 0.6, 0.4, 1.0e-4, 1.1e-4, 0.0, 0.0], dtype=np.float32)
    action = agent.predict(blocked)
    assert action[0] == pytest.approx(0.5)
    assert action[1] == -1.0


def test_action_stays_inside_the_public_contract() -> None:
    agent = RelativeOdorRuleAgent()
    for left, right in ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.3, 0.3)):
        forward, turn = agent.predict(_observation(left, right))
        assert 0.0 <= forward <= 1.0
        assert -1.0 <= turn <= 1.0


def test_requires_exactly_seven_observations() -> None:
    with pytest.raises(ValueError, match="seven-dimensional"):
        RelativeOdorRuleAgent().predict(np.zeros(8, dtype=np.float32))


def test_antenna_inside_a_wall_is_not_trusted() -> None:
    agent = RelativeOdorRuleAgent()
    assert agent.relative_odor_difference(0.0, 0.02) == 0.0
    assert agent.relative_odor_difference(0.02, 0.0) == 0.0
    # Blocked, right antenna inside the wall: fall back to the obstacle rule.
    blocked = np.asarray([0.5, 0.6, 0.4, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
    assert agent.predict(blocked)[1] == 1.0
