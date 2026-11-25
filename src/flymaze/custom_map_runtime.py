"""Adapt validated grid data to the existing FlyGym map interface."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from flymaze.custom_maps import GridMapSpec, read_grid_map
from flymaze.food import food_world_configurator
from flymaze.odor import (
    APPROVED_ODOR_CELL_SIZE_MM,
    APPROVED_ODOR_PARAMETERS,
    StaticOdorField,
    solve_static_odor_field,
)
from flymaze.wall_collision import MAZE_WALL_GEOM_GROUP, add_explicit_wall_pairs

if TYPE_CHECKING:
    from flymaze.maps import MazeMapDefinition


def build_custom_odor_field(spec: GridMapSpec) -> StaticOdorField:
    min_x, max_x, min_y, max_y = spec.bounds_mm
    size = APPROVED_ODOR_CELL_SIZE_MM
    xs = np.arange(min_x, max_x + size * 0.5, size)
    ys = np.arange(min_y, max_y + size * 0.5, size)
    xx, yy = np.meshgrid(xs, ys)
    columns = np.floor(xx / 10 + spec.start[0] + 0.5).astype(int)
    rows = np.floor(yy / 10 + spec.start[1] + 0.5).astype(int)
    mask = np.zeros(xx.shape, dtype=bool)
    for x, y in spec.cells:
        mask |= (columns == x) & (rows == y)
    food_x, food_y = spec.point_mm(spec.food)
    return solve_static_odor_field(
        free_mask=mask,
        source_cell=(int(np.argmin(abs(ys - food_y))), int(np.argmin(abs(xs - food_x)))),
        origin_x_mm=float(xs[0]), origin_y_mm=float(ys[0]), cell_size_mm=size,
        parameters=APPROVED_ODOR_PARAMETERS,
    )


def make_map_definition(spec: GridMapSpec) -> MazeMapDefinition:
    # Delay this import because maps.py calls this adapter after defining its built-ins.
    from flymaze.maps import MazeMapDefinition, PolylineRoute
    from flymaze.topology import MazeEdge, MazeNode, MazeTopology, NodeKind

    cells = sorted(spec.cells)
    ids = {cell: f"cell_{cell[0]}_{cell[1]}" for cell in cells}
    nodes = tuple(
        MazeNode(
            ids[cell], *spec.point_mm(cell),
            NodeKind.START if cell == spec.start else (
                NodeKind.FOOD if cell == spec.food else NodeKind.WAYPOINT
            ),
        )
        for cell in cells
    )
    edges = tuple(
        MazeEdge(ids[(x, y)], ids[neighbor])
        for x, y in cells
        for neighbor in ((x + 1, y), (x, y + 1)) if neighbor in spec.cells
    )
    boxes = spec.wall_boxes()
    names = {f"edge_{i}": f"custom_wall_{i}" for i in range(len(boxes))}

    def configure(world: Any, fly: Any) -> None:
        walls = [
            world.mjcf_root.worldbody.add_geom(
                name=f"custom_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[box.half_x_mm, box.half_y_mm, 1.5],
                pos=[box.x_mm, box.y_mm, 1.5],
                contype=0, conaffinity=0, group=MAZE_WALL_GEOM_GROUP,
                rgba=[0.25, 0.40, 0.48, 1.0],
            )
            for i, box in enumerate(boxes)
        ]
        add_explicit_wall_pairs(world, fly, walls)

    return MazeMapDefinition(
        map_id=spec.map_id, split="train_custom",
        topology=MazeTopology(nodes, edges),
        route=PolylineRoute(tuple(spec.point_mm(cell) for cell in spec.shortest_path())),
        food_xy_mm=spec.point_mm(spec.food), timeout_seconds=spec.timeout_seconds,
        world_configurators=(configure, food_world_configurator(spec.point_mm(spec.food))),
        wall_geom_names_by_side=names, odor_bounds_mm=spec.bounds_mm, custom_grid=spec,
    )


def register_environment_maps(registry: dict[str, MazeMapDefinition]) -> None:
    """Opt-in only. Built-ins and the held-out map are untouched without these variables."""
    filename = os.environ.get("FLYMAZE_MAP_FILE")
    train_id = os.environ.get("FLYMAZE_TRAIN_MAP")
    if not filename:
        if train_id:
            raise ValueError("FLYMAZE_TRAIN_MAP requires FLYMAZE_MAP_FILE")
        return
    path = Path(filename).resolve(strict=True)
    spec = read_grid_map(path)
    definition = make_map_definition(spec)
    if train_id and train_id != spec.map_id:
        raise ValueError("FLYMAZE_TRAIN_MAP must match the custom map_id in FLYMAZE_MAP_FILE")
    registry[spec.map_id] = definition
    if train_id:
        # Existing scratch PPO and its training evaluation both use this registry slot.
        registry["train_l_left"] = definition
    print(
        f"[FlyMaze map] {spec.map_id}: {path} "
        f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}",
        file=sys.stderr,
    )
