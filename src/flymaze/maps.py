"""Versioned MVP map definitions with separated topology, geometry, route, and odor mask."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from flygym.compose import FlatGroundWorld, NeuroMechFly

from flymaze.corridor import CorridorSpec, corridor_world_configurator
from flymaze.custom_maps import GridMapSpec
from flymaze.food import food_world_configurator
from flymaze.geometry import (
    RIGHT_CORNER_WALL_NAMES,
    S_TURN_WALL_NAMES,
    LeftCornerFixtureSpec,
    STurnFixtureSpec,
    left_corner_world_configurator,
    right_corner_world_configurator,
    s_turn_world_configurator,
)
from flymaze.maze_dimensions import STANDARD_CORRIDOR_WIDTH_MM
from flymaze.odor import (
    APPROVED_ODOR_CELL_SIZE_MM,
    APPROVED_ODOR_PARAMETERS,
    StaticOdorField,
    solve_static_odor_field,
)
from flymaze.topology import MazeEdge, MazeNode, MazeTopology, NodeKind

WorldConfigurator = Callable[[FlatGroundWorld, NeuroMechFly], None]


@dataclass(frozen=True, slots=True)
class PolylineRoute:
    waypoints_xy_mm: tuple[tuple[float, float], ...]

    @property
    def total_length_mm(self) -> float:
        points = np.asarray(self.waypoints_xy_mm, dtype=np.float64)
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def remaining_distance_mm(self, point_xy_mm: tuple[float, float]) -> float:
        point = np.asarray(point_xy_mm, dtype=np.float64)
        points = np.asarray(self.waypoints_xy_mm, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        remaining_after = np.concatenate((np.cumsum(segment_lengths[::-1])[::-1], [0.0]))
        candidates: list[float] = []
        for index, (start, end) in enumerate(pairwise(points)):
            delta = end - start
            denominator = float(np.dot(delta, delta))
            fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
            projection = start + fraction * delta
            lateral = float(np.linalg.norm(point - projection))
            along_remaining = (1.0 - fraction) * segment_lengths[index] + remaining_after[index + 1]
            candidates.append(lateral + float(along_remaining))
        return min(candidates)


@dataclass(frozen=True, slots=True)
class MazeMapDefinition:
    map_id: str
    split: str
    topology: MazeTopology
    route: PolylineRoute
    food_xy_mm: tuple[float, float]
    timeout_seconds: float
    world_configurators: tuple[WorldConfigurator, ...]
    wall_geom_names_by_side: dict[str, str]
    odor_bounds_mm: tuple[float, float, float, float]
    custom_grid: GridMapSpec | None = None

    def configure_world(self, world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        for configurator in self.world_configurators:
            configurator(world, fly)

    def build_odor_field(self) -> StaticOdorField:
        if self.custom_grid is not None:
            from flymaze.custom_map_runtime import build_custom_odor_field

            return build_custom_odor_field(self.custom_grid)
        min_x, max_x, min_y, max_y = self.odor_bounds_mm
        cell_size = APPROVED_ODOR_CELL_SIZE_MM
        x_values = np.arange(min_x, max_x + cell_size * 0.5, cell_size)
        y_values = np.arange(min_y, max_y + cell_size * 0.5, cell_size)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        if self.map_id == "train_l_left":
            free_mask = ((y_grid >= -5.0) & (y_grid <= 5.0)) | ((x_grid >= 5.0) & (x_grid <= 15.0))
        elif self.map_id == "train_l_right":
            free_mask = ((y_grid >= -5.0) & (y_grid <= 5.0)) | ((x_grid >= 5.0) & (x_grid <= 15.0))
        elif self.map_id == "test_straight_unknown":
            free_mask = (y_grid >= -5.0) & (y_grid <= 5.0)
        elif self.map_id == "validation_s_turn_complex":
            free_mask = (
                ((x_grid >= -5.0) & (x_grid <= 15.0) & (y_grid >= -5.0) & (y_grid <= 5.0))
                | ((x_grid >= 5.0) & (x_grid <= 15.0) & (y_grid >= -5.0) & (y_grid <= 20.0))
                | ((x_grid >= 5.0) & (x_grid <= 30.0) & (y_grid >= 10.0) & (y_grid <= 20.0))
            )
        else:
            raise ValueError(f"odor mask is not defined for map: {self.map_id}")
        source_column = int(np.argmin(np.abs(x_values - self.food_xy_mm[0])))
        source_row = int(np.argmin(np.abs(y_values - self.food_xy_mm[1])))
        return solve_static_odor_field(
            free_mask=free_mask,
            source_cell=(source_row, source_column),
            origin_x_mm=float(x_values[0]),
            origin_y_mm=float(y_values[0]),
            cell_size_mm=cell_size,
            parameters=APPROVED_ODOR_PARAMETERS,
        )


TRAIN_L_LEFT = MazeMapDefinition(
    map_id="train_l_left",
    split="train",
    topology=MazeTopology(
        nodes=(
            MazeNode("start", 0.0, 0.0, NodeKind.START),
            MazeNode("corner", 10.0, 0.0, NodeKind.WAYPOINT),
            MazeNode("food", 10.0, 18.0, NodeKind.FOOD),
        ),
        edges=(MazeEdge("start", "corner"), MazeEdge("corner", "food")),
    ),
    route=PolylineRoute(((0.0, 0.0), (10.0, 0.0), (10.0, 18.0))),
    food_xy_mm=(10.0, 18.0),
    timeout_seconds=10.0,
    world_configurators=(
        left_corner_world_configurator(LeftCornerFixtureSpec()),
        food_world_configurator((10.0, 18.0)),
    ),
    wall_geom_names_by_side={
        "outer_bottom": "corner_outer_bottom",
        "inner_horizontal": "corner_inner_horizontal",
        "inner_vertical": "corner_inner_vertical",
        "outer_vertical": "corner_outer_vertical",
    },
    odor_bounds_mm=(-5.0, 15.0, -5.0, 20.0),
)


TRAIN_L_RIGHT = MazeMapDefinition(
    map_id="train_l_right",
    split="train",
    topology=MazeTopology(
        nodes=(
            MazeNode("start", 0.0, 0.0, NodeKind.START),
            MazeNode("corner", 10.0, 0.0, NodeKind.WAYPOINT),
            MazeNode("food", 10.0, -18.0, NodeKind.FOOD),
        ),
        edges=(MazeEdge("start", "corner"), MazeEdge("corner", "food")),
    ),
    route=PolylineRoute(((0.0, 0.0), (10.0, 0.0), (10.0, -18.0))),
    food_xy_mm=(10.0, -18.0),
    timeout_seconds=10.0,
    world_configurators=(
        right_corner_world_configurator(LeftCornerFixtureSpec()),
        food_world_configurator((10.0, -18.0)),
    ),
    wall_geom_names_by_side=RIGHT_CORNER_WALL_NAMES,
    odor_bounds_mm=(-5.0, 15.0, -20.0, 5.0),
)


TEST_STRAIGHT_UNKNOWN = MazeMapDefinition(
    map_id="test_straight_unknown",
    split="held_out_test",
    topology=MazeTopology(
        nodes=(
            MazeNode("start", 0.0, 0.0, NodeKind.START),
            MazeNode("food", 18.0, 0.0, NodeKind.FOOD),
        ),
        edges=(MazeEdge("start", "food"),),
    ),
    route=PolylineRoute(((0.0, 0.0), (18.0, 0.0))),
    food_xy_mm=(18.0, 0.0),
    timeout_seconds=8.0,
    world_configurators=(
        corridor_world_configurator(
            CorridorSpec(
                width_mm=STANDARD_CORRIDOR_WIDTH_MM,
                length_mm=30.0,
                center_x_mm=12.0,
            )
        ),
        food_world_configurator((18.0, 0.0)),
    ),
    wall_geom_names_by_side={
        "left": "corridor_left_wall",
        "right": "corridor_right_wall",
    },
    odor_bounds_mm=(-5.0, 22.0, -5.0, 5.0),
)


VALIDATION_S_TURN_COMPLEX = MazeMapDefinition(
    map_id="validation_s_turn_complex",
    split="train_auxiliary",
    topology=MazeTopology(
        nodes=(
            MazeNode("start", 0.0, 0.0, NodeKind.START),
            MazeNode("left_corner", 10.0, 0.0, NodeKind.WAYPOINT),
            MazeNode("right_corner", 10.0, 15.0, NodeKind.WAYPOINT),
            MazeNode("food", 25.0, 15.0, NodeKind.FOOD),
        ),
        edges=(
            MazeEdge("start", "left_corner"),
            MazeEdge("left_corner", "right_corner"),
            MazeEdge("right_corner", "food"),
        ),
    ),
    route=PolylineRoute(((0.0, 0.0), (10.0, 0.0), (10.0, 15.0), (25.0, 15.0))),
    food_xy_mm=(25.0, 15.0),
    timeout_seconds=14.0,
    world_configurators=(
        s_turn_world_configurator(STurnFixtureSpec()),
        food_world_configurator((25.0, 15.0)),
    ),
    wall_geom_names_by_side=S_TURN_WALL_NAMES,
    odor_bounds_mm=(-5.0, 30.0, -5.0, 20.0),
)


MAPS = {
    definition.map_id: definition
    for definition in (
        TRAIN_L_LEFT,
        TRAIN_L_RIGHT,
        TEST_STRAIGHT_UNKNOWN,
        VALIDATION_S_TURN_COMPLEX,
    )
}


# Custom maps are opt-in; the original registry is identical without environment flags.
def _load_custom_maps() -> None:
    import os

    if os.environ.get("FLYMAZE_MAP_FILE") or os.environ.get("FLYMAZE_TRAIN_MAP"):
        from flymaze.custom_map_runtime import register_environment_maps

        register_environment_maps(MAPS)


_load_custom_maps()
