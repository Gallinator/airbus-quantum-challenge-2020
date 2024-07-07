import numpy as np

from utils import get_linear_shear_curve


class AircraftData:
    def __init__(self, num_positions, payload_area_length, max_payload, shear_curve, min_cg, max_cg):
        self.num_positions = num_positions
        self.payload_area_length = payload_area_length
        self.max_payload = max_payload
        if not self.is_shear_curve_valid(shear_curve):
            raise ValueError(f'Invalid shear curve length of {len(shear_curve)}')
        self.shear_curve = shear_curve
        self.min_cg = min_cg * payload_area_length
        self.max_cg = max_cg * payload_area_length
        self.locations = self._get_locations()
        self.location_ends = self._get_location_ends()

        if min_cg > max_cg:
            raise ValueError(f'Invalid cg limits ({min_cg}>{max_cg})')

    def _get_locations(self) -> np.ndarray:
        pos = []
        for i in range(self.num_positions):
            x_i = (self.payload_area_length / self.num_positions *
                   ((1 + i) - self.num_positions / 2) -
                   self.payload_area_length / (2 * self.num_positions))
            pos.append(x_i)
        return np.array(pos)

    def _get_location_ends(self) -> np.ndarray:
        pos = []
        for u in range(self.num_positions):
            x_u = self.payload_area_length / self.num_positions * ((u + 1) - self.num_positions / 2)
            pos.append(x_u)
        return np.array(pos)

    def is_shear_curve_valid(self, shear_curve):
        is_even_pos = self.num_positions % 2 == 0
        if is_even_pos:
            # Take into account center left and right shear
            return len(shear_curve) == self.num_positions + 1
        else:
            # The center point is not accounted for in num_positions
            return len(shear_curve) == self.num_positions + 2


def get_default_aircraft() -> AircraftData:
    shear_curve = get_linear_shear_curve(20, 26000)
    return AircraftData(20, 40, 40000, shear_curve, -0.1, 0.2)
