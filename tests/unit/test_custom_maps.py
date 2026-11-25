"""Map authoring checks runnable without FlyGym using unittest discovery."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flymaze.custom_maps import GridMapSpec, read_grid_map


def example() -> dict[str, Any]:
    return {
        "version": 1, "map_id": "custom_left", "width": 6, "height": 5,
        "cell_size_mm": 10, "timeout_seconds": 14,
        "start": [1, 1], "food": [3, 3],
        "cells": [[1, 1], [2, 1], [3, 1], [3, 2], [3, 3]],
    }


class CustomMapTests(unittest.TestCase):
    def test_coordinates_route_and_serialization(self) -> None:
        spec = GridMapSpec.from_dict(example())
        self.assertEqual(spec.point_mm(spec.start), (0.0, 0.0))
        self.assertEqual(spec.point_mm(spec.food), (20.0, 20.0))
        self.assertEqual(len(spec.shortest_path()), 5)
        self.assertEqual(GridMapSpec.from_dict(spec.to_dict()), spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
            self.assertEqual(read_grid_map(path), spec)

    def test_rejects_invalid_and_unsafe_maps(self) -> None:
        changes = [
            {"map_id": "test_straight_unknown"}, {"map_id": "../custom_bad"},
            {"version": 2}, {"cell_size_mm": 9}, {"timeout_seconds": float("nan")},
            {"timeout_seconds": True}, {"width": 3.5}, {"start": [1, True]},
            {"food": [1, 1]}, {"food": [5, 4]}, {"width": 10000},
            {"cells": [[1, 1], [3, 3]]},
            {"cells": example()["cells"] + [[5, 4]]},
            {"cells": example()["cells"] + [[1, 1]]},
            {"cells": example()["cells"] + [[2, 2]]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                GridMapSpec.from_dict(example() | change)

    def test_requires_eastward_spawn_clearance(self) -> None:
        raw = example()
        raw["cells"] = [[1, 1], [1, 2], [1, 3]]
        raw["food"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "east"):
            GridMapSpec.from_dict(raw)

    def test_walls_enclose_only_exposed_cell_edges(self) -> None:
        spec = GridMapSpec.from_dict(example())
        walls = spec.wall_boxes()
        # Five cells have 20 edges, four shared edges remove eight boundaries.
        total_length = sum(2 * max(w.half_x_mm, w.half_y_mm) for w in walls)
        self.assertEqual(total_length, 120.0)
        for cell in spec.cells:
            x, y = spec.point_mm(cell)
            self.assertTrue(spec.is_free(x, y))
            for wall in walls:
                self.assertFalse(
                    abs(x - wall.x_mm) < wall.half_x_mm
                    and abs(y - wall.y_mm) < wall.half_y_mm
                )
        self.assertFalse(spec.is_free(0, 20))
        self.assertTrue(spec.is_free(10, 0))

    def test_branches_have_a_shortest_connected_route(self) -> None:
        raw = copy.deepcopy(example())
        raw["cells"] += [[2, 0]]
        spec = GridMapSpec.from_dict(raw)
        self.assertEqual(spec.shortest_path()[0], spec.start)
        self.assertEqual(spec.shortest_path()[-1], spec.food)
        self.assertEqual(len(spec.shortest_path()), 5)


if __name__ == "__main__":
    unittest.main()
