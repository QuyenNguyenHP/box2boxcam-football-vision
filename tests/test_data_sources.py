from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pitchlens.data.list_sources import load_sources


class DataSourceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ROOT / "configs/data_sources.toml"
        self.sources = load_sources(self.registry)

    def test_registry_is_valid_toml(self) -> None:
        with self.registry.open("rb") as stream:
            self.assertIn("sources", tomllib.load(stream))

    def test_source_ids_are_unique(self) -> None:
        identifiers = [source["id"] for source in self.sources]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_unknown_licenses_are_blocked(self) -> None:
        for source in self.sources:
            if source["data_license"] == "unknown":
                self.assertTrue(str(source["status"]).startswith("blocked"))


if __name__ == "__main__":
    unittest.main()
