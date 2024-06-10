import unittest

from coefficient_tuning import get_oom


class CoefficientTuningTest(unittest.TestCase):
    def test_zero_oom(self):
        v = 0

        self.assertEqual(0, get_oom(0))
