import unittest

import numpy as np

from aircraft_data import get_default_aircraft


class AircraftDataTests(unittest.TestCase):
    def test_locations(self):
        acft = get_default_aircraft()

        locations = np.array([-19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

        self.assertEqual(acft.locations.all(), locations.all())
