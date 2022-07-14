# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from examples.few_shot.scripts.significance_test import ProportionSignificanceTest


class TestProportionSignificanceTest(unittest.TestCase):
    def test_sig_level_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, n2=1000, p1=0.70, p2=0.75, power=0.9, sig_level=None
        )
        self.assertAlmostEqual(sig_level, 0.2205, 3)

        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1673, n2=1673, p1=0.70, p2=0.75, power=0.9, sig_level=None
        )
        self.assertAlmostEqual(sig_level, 0.05, 3)

    def test_sig_level_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000,
            n2=1000,
            p1=0.70,
            p2=0.75,
            power=0.9,
            sig_level=None,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.1103, 3)

        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1364,
            n2=1364,
            p1=0.70,
            p2=0.75,
            power=0.9,
            sig_level=None,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.05, 3)

    def test_power_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, n2=1000, p1=0.70, p2=0.75, sig_level=0.05, power=None
        )
        self.assertAlmostEqual(sig_level, 0.7076, 3)

        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1430, n2=1430, p1=0.70, p2=0.75, sig_level=0.05, power=None
        )
        self.assertAlmostEqual(sig_level, 0.85, 3)

    def test_power_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000,
            n2=1000,
            p1=0.70,
            p2=0.75,
            sig_level=0.05,
            power=None,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.8055, 3)

        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1145,
            n2=1145,
            p1=0.70,
            p2=0.75,
            sig_level=0.05,
            power=None,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.85, 3)

    def test_sample_size_for_equal_size_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            p1=0.70, p2=0.75, power=0.9, sig_level=0.05
        )
        self.assertAlmostEqual(sig_level, 1672.8418, 3)

    def test_sample_size_for_unequal_size_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, p1=0.70, p2=0.75, power=0.9, sig_level=0.05
        )
        self.assertAlmostEqual(sig_level, 5113.2500, 3)

    def test_sample_size_for_equal_size_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            p1=0.70, p2=0.75, power=0.9, sig_level=0.05, alternative="smaller"
        )
        self.assertAlmostEqual(sig_level, 1363.4139, 3)

    def test_sample_size_for_unequal_size_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, p1=0.70, p2=0.75, power=0.9, sig_level=0.05, alternative="smaller"
        )
        self.assertAlmostEqual(sig_level, 2141.7589, 3)

    def test_p1_for_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, n2=1000, p1=None, p2=0.75, power=0.9, sig_level=0.05
        )
        self.assertAlmostEqual(sig_level, 0.8099, 3)

    def test_p2_for_2_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000, n2=1000, p1=0.70, p2=None, power=0.9, sig_level=0.05
        )
        self.assertAlmostEqual(sig_level, 0.6317, 3)

    def test_p1_for_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000,
            n2=1000,
            p1=None,
            p2=0.75,
            power=0.9,
            sig_level=0.05,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.6914, 3)

    def test_p2_for_1_sided(self):
        sig_level, suggestions = ProportionSignificanceTest().solve_2sample(
            n1=1000,
            n2=1000,
            p1=0.70,
            p2=None,
            power=0.9,
            sig_level=0.05,
            alternative="smaller",
        )
        self.assertAlmostEqual(sig_level, 0.7581, 3)


if __name__ == "__main__":
    unittest.main()
