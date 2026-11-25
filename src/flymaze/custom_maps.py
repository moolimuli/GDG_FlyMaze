"""Validated, portable map data. This module needs only Python's standard library."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flymaze.maze_dimensions import STANDARD_CORRIDOR_WIDTH_MM

Cell = tuple[int, int]
DIRECTIONS: tuple[Cell, ...] = ((1, 0), (0, 1), (0, -1), (-1, 0))


@dataclass(frozen=True)
class WallBox:
    x_mm: float
    y_mm: float
    half_x_mm: float
    half_y_mm: float


@dataclass(frozen=True)
class GridMapSpec:
    map_id: str
    width: int
    height: int
    start: Cell
    food: Cell
    cells: frozenset[Cell]
    timeout_seconds: float

    @classmethod
    def from_dict(cls, raw: Any) -> GridMapSpec:
        if not isinstance(raw, dict):
            raise ValueError("map must be a JSON object")
        if type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("version must be 1")
        map_id = raw.get("map_id")
        if not isinstance(map_id, str) or not re.fullmatch(r"custom_[a-z0-9_]{1,48}", map_id):
            raise ValueError("map_id must be custom_ followed by 1-48 lowercase letters/digits/_")
        width, height = raw.get("width"), raw.get("height")
        if any(type(v) is not int or not 3 <= v <= 12 for v in (width, height)):
            raise ValueError("width and height must be integers from 3 to 12")
        if type(raw.get("cell_size_mm")) not in (int, float):
            raise ValueError("cell_size_mm must be 10")
        if raw["cell_size_mm"] != STANDARD_CORRIDOR_WIDTH_MM:
            raise ValueError("cell_size_mm must be 10 (fixed standard corridor width)")
        timeout = raw.get("timeout_seconds")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 1 <= timeout <= 300
        ):
            raise ValueError("timeout_seconds must be finite and between 1 and 300")

        def cell(value: Any) -> Cell:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("each cell must be [column, row]")
            if any(type(v) is not int for v in value):
                raise ValueError("cell coordinates must be integers")
            x, y = value
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("cell is outside the map")
            return x, y

        start, food = cell(raw.get("start")), cell(raw.get("food"))
        values = raw.get("cells")
        if not isinstance(values, list) or not 2 <= len(values) <= width * height:
            raise ValueError("cells must contain 2 to width*height walkable cells")
        cells = frozenset(cell(value) for value in values)
        if len(cells) != len(values):
            raise ValueError("duplicate cells are not allowed")
        if start == food or start not in cells or food not in cells:
            raise ValueError("start and food must be different walkable cells")
        if (start[0] + 1, start[1]) not in cells:
            raise ValueError("the cell east of start must be free for the fixed +X spawn heading")
        if any(
            {(x + 1, y), (x, y + 1), (x + 1, y + 1)} <= cells for x, y in cells
        ):
            raise ValueError("2x2 open rooms are not supported; keep corridors one cell wide")
        spec = cls(map_id, width, height, start, food, cells, float(timeout))
        parents = spec._parents()
        if len(parents) != len(cells):
            raise ValueError("all walkable cells must connect to start using shared edges")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "map_id": self.map_id, "width": self.width, "height": self.height,
            "cell_size_mm": STANDARD_CORRIDOR_WIDTH_MM,
            "start": list(self.start), "food": list(self.food),
            "cells": [list(cell) for cell in sorted(self.cells)],
            "timeout_seconds": self.timeout_seconds,
        }

    def point_mm(self, cell: Cell) -> tuple[float, float]:
        size = STANDARD_CORRIDOR_WIDTH_MM
        return (cell[0] - self.start[0]) * size, (cell[1] - self.start[1]) * size

    def _parents(self) -> dict[Cell, Cell | None]:
        parents: dict[Cell, Cell | None] = {self.start: None}
        queue = deque([self.start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                neighbor = x + dx, y + dy
                if neighbor in self.cells and neighbor not in parents:
                    parents[neighbor] = (x, y)
                    queue.append(neighbor)
        return parents

    def shortest_path(self) -> tuple[Cell, ...]:
        parents = self._parents()
        current: Cell | None = self.food
        result = []
        while current is not None:
            result.append(current)
            current = parents[current]
        return tuple(reversed(result))

    def is_free(self, x_mm: float, y_mm: float) -> bool:
        size = STANDARD_CORRIDOR_WIDTH_MM
        x = math.floor(x_mm / size + self.start[0] + 0.5)
        y = math.floor(y_mm / size + self.start[1] + 0.5)
        return (x, y) in self.cells

    @property
    def bounds_mm(self) -> tuple[float, float, float, float]:
        points = [self.point_mm(cell) for cell in self.cells]
        return (
            min(p[0] for p in points) - 5, max(p[0] for p in points) + 5,
            min(p[1] for p in points) - 5, max(p[1] for p in points) + 5,
        )

    def wall_boxes(self) -> tuple[WallBox, ...]:
        """Merge exposed collinear edges; place 1 mm walls outside free space."""
        groups: dict[tuple[str, float], list[float]] = defaultdict(list)
        for cell in sorted(self.cells):
            x, y = self.point_mm(cell)
            for dx, dy in DIRECTIONS:
                if (cell[0] + dx, cell[1] + dy) in self.cells:
                    continue
                if dx:
                    groups[("v", x + dx * 5.5)].append(y)
                else:
                    groups[("h", y + dy * 5.5)].append(x)
        walls = []
        for (orientation, fixed), centers in sorted(groups.items()):
            centers.sort()
            first = previous = centers[0]
            for value in [*centers[1:], math.inf]:
                if value != previous + 10:
                    middle = (first + previous) / 2
                    half_length = (previous - first + 10) / 2
                    if orientation == "h":
                        walls.append(WallBox(middle, fixed, half_length, 0.5))
                    else:
                        walls.append(WallBox(fixed, middle, 0.5, half_length))
                    first = value
                previous = value
        return tuple(walls)


def read_grid_map(path: str | Path) -> GridMapSpec:
    path = Path(path)
    if path.stat().st_size > 100_000:
        raise ValueError("map JSON exceeds 100 KB")
    return GridMapSpec.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
