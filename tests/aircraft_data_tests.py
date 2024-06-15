import unittest

import numpy as np

from aircraft_data import get_default_aircraft


class AircraftDataTests(unittest.TestCase):
    def test_locations(self):
        acft = get_default_aircraft()

        locations = np.array([-19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

        self.assertEqual(acft.locations.all(), locations.all())

    def test_location_ends(self):
        acft = get_default_aircraft()

        locations = np.array([-18, -16, 14, -12, -10, -8, -6, 4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

        self.assertEqual(acft.location_ends.all(), locations.all())
