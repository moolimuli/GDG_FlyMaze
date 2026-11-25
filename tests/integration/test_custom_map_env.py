"""Run in the project's Ubuntu uv environment with FlyGym installed."""

import json
from pathlib import Path

import numpy as np
import pytest

from flymaze.custom_map_runtime import make_map_definition, register_environment_maps
from flymaze.custom_maps import GridMapSpec, read_grid_map
from flymaze.env import FlyMazeEnv
from flymaze.maps import MAPS

RAW = {
    "version": 1, "map_id": "custom_smoke", "width": 5, "height": 4,
    "cell_size_mm": 10, "timeout_seconds": 14,
    "start": [1, 1], "food": [3, 2],
    "cells": [[1, 1], [2, 1], [3, 1], [3, 2]],
}


def test_custom_odor_source_and_walls() -> None:
    definition = make_map_definition(GridMapSpec.from_dict(RAW))
    field = definition.build_odor_field()
    assert field.sample(*definition.food_xy_mm) == pytest.approx(1.0)
    assert field.sample(0, 10) == 0.0
    assert field.sample(10, 0) > 0
    assert definition.route.total_length_mm == pytest.approx(30.0)
    assert definition.topology.start.x_mm == 0.0
    assert definition.topology.start.y_mm == 0.0


@pytest.mark.parametrize("name", ["custom_race_a", "custom_raceb"])
def test_shared_map_odor_source_and_route(name: str) -> None:
    directory = Path(__file__).resolve().parents[2] / "maps"
    spec = read_grid_map(directory / f"{name}.json")
    definition = make_map_definition(spec)
    field = definition.build_odor_field()
    assert field.sample(*definition.food_xy_mm) == pytest.approx(1.0)
    assert np.isfinite(field.normalized_concentration).all()
    assert definition.route.total_length_mm == pytest.approx(
        (len(spec.shortest_path()) - 1) * 10
    )
    for cell in spec.cells:
        x, y = spec.point_mm(cell)
        row = round((y - field.origin_y_mm) / field.cell_size_mm)
        column = round((x - field.origin_x_mm) / field.cell_size_mm)
        assert field.free_mask[row, column]


def test_opt_in_registry_preserves_held_out_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = dict(MAPS)
    monkeypatch.delenv("FLYMAZE_MAP_FILE", raising=False)
    monkeypatch.delenv("FLYMAZE_TRAIN_MAP", raising=False)
    registry = dict(original)
    register_environment_maps(registry)
    assert registry == original
    path = tmp_path / "custom_smoke.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    monkeypatch.setenv("FLYMAZE_MAP_FILE", str(path))
    register_environment_maps(registry)
    assert registry["train_l_left"] is original["train_l_left"]
    monkeypatch.setenv("FLYMAZE_TRAIN_MAP", "custom_smoke")
    register_environment_maps(registry)
    assert registry["train_l_left"].map_id == "custom_smoke"
    assert registry["test_straight_unknown"] is original["test_straight_unknown"]
    monkeypatch.setenv("FLYMAZE_TRAIN_MAP", "test_straight_unknown")
    with pytest.raises(ValueError):
        register_environment_maps(registry)


@pytest.mark.integration
def test_custom_map_headless_reset_step(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = make_map_definition(GridMapSpec.from_dict(RAW))
    monkeypatch.setitem(MAPS, definition.map_id, definition)
    env = FlyMazeEnv(definition.map_id)
    try:
        observation, info = env.reset(seed=0)
        assert observation.shape == (7,)
        assert info["map_id"] == "custom_smoke"
        observation, reward, _, _, _ = env.step(np.asarray([0.1, 0.0], dtype=np.float32))
        assert observation.shape == (7,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
    finally:
        env.close()
