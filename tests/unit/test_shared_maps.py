"""Regression coverage for the two supplied JSON maps and their original settings."""

import unittest
from pathlib import Path

from flymaze.custom_maps import read_grid_map

MAP_DIRECTORY = Path(__file__).resolve().parents[2] / "maps"


class SharedMapTests(unittest.TestCase):
    def test_shared_maps_preserve_ids_dimensions_and_timeout(self) -> None:
        for name, width, height, count, route_mm in (
            ("custom_race_a", 8, 6, 12, 110),
            ("custom_raceb", 10, 8, 45, 220),
        ):
            with self.subTest(name=name):
                spec = read_grid_map(MAP_DIRECTORY / f"{name}.json")
                self.assertEqual(spec.map_id, name)
                self.assertEqual((spec.width, spec.height), (width, height))
                self.assertEqual(len(spec.cells), count)
                self.assertEqual(spec.timeout_seconds, 14.0)
                self.assertEqual((len(spec.shortest_path()) - 1) * 10, route_mm)

    def test_both_food_sources_are_reachable_from_the_east_facing_spawn(self) -> None:
        for name in ("custom_race_a", "custom_raceb"):
            with self.subTest(name=name):
                spec = read_grid_map(MAP_DIRECTORY / f"{name}.json")
                route = spec.shortest_path()
                self.assertEqual(route[0], spec.start)
                self.assertEqual(route[-1], spec.food)
                self.assertIn((spec.start[0] + 1, spec.start[1]), spec.cells)


if __name__ == "__main__":
    unittest.main()
