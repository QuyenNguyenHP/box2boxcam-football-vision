"""Checks for the match-level split used by the private Luna pilot."""

import unittest
from collections import Counter

from box2boxcam.data.build_luna import choose_splits


class MatchSplitsTest(unittest.TestCase):
    def test_deterministic_and_exclusive(self):
        groups = {
            f"match_{index}": Counter(images=100 + index, positive=75, negative=25 + index)
            for index in range(10)
        }
        first = choose_splits(groups)
        self.assertEqual(first, choose_splits(groups))
        self.assertEqual(set(first), set(groups))
        self.assertEqual(Counter(first.values()), Counter(train=6, val=2, test=2))

    def test_too_few_matches(self):
        with self.assertRaises(ValueError):
            choose_splits({"only_match": Counter(images=10, positive=10)})


if __name__ == "__main__":
    unittest.main()
