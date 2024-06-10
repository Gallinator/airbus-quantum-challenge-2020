import unittest
import numpy as np
from utils import get_container_t, get_container_d


class UtilsTests(unittest.TestCase):

    def test_container_t(self):
        cont_types = np.array(['t1', 't2', 't3'])

        cont_t = get_container_t(cont_types)
        true_t = np.array([1, 1, 1 / 2])

        self.assertEqual(cont_t.all(), true_t.all())

    def test_container_d(self):
        cont_types = np.array(['t1', 't2', 't3'])

        cont_d = get_container_d(cont_types)
        true_d = np.array([1, 1 / 2, 1])

        self.assertEqual(cont_d.all(), true_d.all())
