import numpy as np


class AircraftData:
    def __init__(self, num_positions, payload_area_length, max_payload, max_shear, min_cg, max_cg):
        self.num_positions = num_positions
        self.payload_area_length = payload_area_length
        self.max_payload = max_payload
        self.max_shear = max_shear
        self.min_cg = min_cg
        self.max_cg = max_cg
        self.locations = self.get_locations()

        if min_cg > max_cg:
            raise ValueError(f'Invalid cg limits ({min_cg}>{max_cg})')

    def get_locations(self) -> np.ndarray:
        pos = []
        for i in range(self.num_positions):
            x_i = (self.payload_area_length / self.num_positions *
                   ((1 + i) - self.num_positions / 2) -
                   self.payload_area_length / (2 * self.num_positions))
            pos.append(x_i)
        return np.array(pos)


def get_default_aircraft() -> AircraftData:
    return AircraftData(20, 40, 40000, 26000, -0.1, 0.2)
