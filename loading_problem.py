import math
from functools import reduce
import numpy as np
from dimod import SampleSet, BQM, Binaries, Binary

from aircraft_data import AircraftData
from utils import get_container_t, get_container_d


class LoadingProblem:
    def __init__(self, aircraft: AircraftData,
                 container_types: np.ndarray,
                 container_masses: np.ndarray,
                 target_cg,
                 zero_payload_mass,
                 zero_payload_cg):

        if len(container_types) != len(container_masses):
            raise ValueError(
                f'Mismatching container types and masses lengths ({len(container_types)},{len(container_masses)})')

        self.aircraft = aircraft
        self.container_types = container_types
        self.container_masses = container_masses
        self.target_cg = target_cg
        self.zero_payload_mass = zero_payload_mass
        self.zero_payload_cg = zero_payload_cg * aircraft.payload_area_length
        self.container_t = get_container_t(container_types)
        self.container_d = get_container_d(container_types)
        self.coefficients = {'pl_o': 1.0, 'pl_w': 1.0, 'pl_d': 1.0, 'pl_c': 1.0}
        self.num_slack_variables = get_num_slack_vars(aircraft, len(container_types))

    def get_objective_bqm(self) -> BQM:
        bqm = BQM.empty('BINARY')
        for i, t_i in enumerate(self.container_t):
            for pos in range(self.aircraft.num_positions):
                bqm += Binary(f'p_{i}_{pos}', -t_i * self.container_masses[i])
        return bqm

    def get_no_overlaps_bqm(self) -> BQM:
        num_slack = self.num_slack_variables['pl_o'] // self.aircraft.num_positions
        bqm = BQM.empty('BINARY')

        for pos in range(self.aircraft.num_positions):
            bqm_pos = BQM.empty('BINARY')
            for i, d_i in enumerate(self.container_d):
                bqm_pos += Binary(f'p_{i}_{pos}', d_i)
            for k in range(num_slack):
                bqm_pos += Binary(f'v_o_{pos}_{k}', 2 ** k)
            bqm_pos += -1
            bqm_pos = bqm_pos ** 2
            bqm += bqm_pos

        bqm.scale(self.coefficients['pl_o'])
        return bqm

    def get_no_duplicates_bqm(self) -> BQM:
        num_slack = self.num_slack_variables['pl_d'] // len(self.container_types)
        bqm = BQM.empty('BINARY')

        for c, t_i in enumerate(self.container_t):
            cont_bqm = BQM.empty('BINARY')
            # Quadratic terms
            for pos in range(self.aircraft.num_positions):
                cont_bqm += Binary(f'p_{c}_{pos}', t_i)
            for k in range(num_slack):
                cont_bqm += Binary(f'v_d_{c}_{k}', 2 ** k)
            cont_bqm += -1
            cont_bqm = cont_bqm ** 2
            bqm += cont_bqm

        bqm.scale(self.coefficients['pl_d'])
        return bqm

    def get_contiguity_bqm(self) -> BQM:
        bqm = BQM.empty('BINARY')

        for i, t_i in enumerate(self.container_t):
            cont_bqm = BQM.empty('BINARY')
            # Not a type 3 container
            if t_i >= 1:
                continue
            else:
                for pos in range(self.aircraft.num_positions - 1):
                    actual_p = Binary(f'p_{i}_{pos}')
                    next_p = Binary(f'p_{i}_{pos + 1}')
                    cont_bqm += 1 / 2 * actual_p - (actual_p * next_p)
                cont_bqm += Binary(f'p_{i}_{self.aircraft.num_positions - 1}', 1 / 2)
                bqm += cont_bqm

        bqm.scale(self.coefficients['pl_c'])
        return bqm

    def get_max_capacity_bqm(self) -> BQM:
        num_slack = self.num_slack_variables['pl_w']
        bqm = BQM.empty('BINARY')
        for i, t_i in enumerate(self.container_t):
            for pos in range(self.aircraft.num_positions):
                coef = t_i * self.container_masses[i]
                bqm += Binary(f'p_{i}_{pos}', coef)
        for k in range(num_slack):
            bqm += Binary(f'v_w_{k}', 2 ** k)
        bqm += -self.aircraft.max_payload
        bqm = bqm ** 2
        bqm.scale(self.coefficients['pl_w'])
        return bqm

    def get_q(self) -> BQM:
        obj_q = self.get_objective_bqm()
        no_overlaps_q = self.get_no_overlaps_bqm()
        no_duplicates_q = self.get_no_duplicates_bqm()
        max_capacity_q = self.get_max_capacity_bqm()
        contiguity_q = self.get_contiguity_bqm()
        return obj_q + no_overlaps_q + no_duplicates_q + max_capacity_q + contiguity_q

    def parse_solution(self, results: SampleSet) -> np.ndarray:
        solutions = []
        for r in results.samples():
            cont_occ = np.zeros(shape=(len(self.container_types), self.aircraft.num_positions))
            for k, v in r.items():
                if k[0] == 'p':
                    i = int(k[2])
                    j = int(k[4])
                    cont_occ[i, j] = v
            solutions.append(cont_occ)
        return np.array(solutions)

    def filter_solutions(self, solutions: np.ndarray) -> list:
        res = []
        for s in solutions:
            if (self.check_overlap_constraint(s) and
                    self.check_no_duplicates_constraint(s) and
                    self.check_max_weight_constraint(s) and
                    self.check_contiguity_constraint(s)):
                res.append(s)
        return res

    def check_overlap_constraint(self, cont_occ: np.ndarray) -> bool:
        for pos in range(self.aircraft.num_positions):
            pos_sum = 0
            for i, d_i in enumerate(self.container_d):
                pos_sum += d_i * cont_occ[i, pos]
            if pos_sum > 1:
                return False
        return True

    def check_contiguity_constraint(self, cont_occ: np.ndarray) -> bool:
        num_cont = cont_occ.shape[0]
        for c in range(num_cont):
            if self.container_types[c] != 't3':
                continue
            row_sum = np.sum(cont_occ[c])
            if row_sum == 0:
                continue
            if row_sum != 2:
                return False
            s = 0
            for pos in range(self.aircraft.num_positions - 1):
                s += cont_occ[c, pos] * cont_occ[c, pos + 1]
            if s != 1:
                return False

        return True

    def check_max_weight_constraint(self, cont_occ: np.ndarray) -> bool:
        weight = 0
        for index, v in np.ndenumerate(cont_occ):
            i, j = index
            weight += self.container_t[i] * self.container_masses[i] * v
        return weight <= self.aircraft.max_payload

    def check_no_duplicates_constraint(self, cont_occ: np.ndarray) -> bool:
        num_cont = cont_occ.shape[0]
        for c in range(num_cont):
            s = 0
            for pos in range(self.aircraft.num_positions):
                s += self.container_t[c] * cont_occ[c, pos]
            if s > 1:
                return False
        return True

    def get_payload_weight(self, cont_occ: np.ndarray) -> float:
        pl_weight = 0
        for index, v in np.ndenumerate(cont_occ):
            c, pos = index
            pl_weight += self.container_t[c] * self.container_masses[c] * v
        return pl_weight

    def get_cg(self, cont_occ: np.ndarray) -> float:
        s1 = 0
        s2 = 0
        for index, p in np.ndenumerate(cont_occ):
            i, j = index
            s2 += self.container_t[i] * self.container_masses[i] * p
            s1 += s2 * self.aircraft.locations[j]
        s1 += self.zero_payload_mass * self.zero_payload_cg
        s2 += self.zero_payload_mass
        return s1 / s2

    def get_shear(self, cont_occ: np.ndarray) -> tuple:
        shear_l = [0]
        shear_r = []

        for u, x_u in enumerate(self.aircraft.location_ends):
            if x_u <= 0:
                shear_l.append(self.get_shear_at_left(u, cont_occ))
            if x_u >= 0:
                shear_r.append(self.get_shear_at_right(u, cont_occ))
        # Odd number of positions
        if self.aircraft.num_positions % 2 != 0:
            s = 0
            j = math.floor(self.aircraft.num_positions / 2.0) + 1
            for i, t_i in enumerate(self.container_t):
                s += t_i * self.container_masses[i] * cont_occ[i, j] / 2.0
            shear_r = [s + shear_l[-1]] + shear_r
            shear_l += [s + shear_r[0]]

        return np.array(shear_l), np.array(shear_r)

    def get_shear_at_left(self, pos, cont_occ: np.ndarray):
        shear = 0
        for i, t_i in enumerate(self.container_t):
            for j in range(pos):
                shear += t_i * self.container_masses[i] * cont_occ[i, j]
        return shear

    def get_shear_at_right(self, pos, cont_occ: np.ndarray):
        shear = 0
        for i, t_i in enumerate(self.container_t):
            for j in range(pos + 1, self.aircraft.num_positions):
                shear += t_i * self.container_masses[i] * cont_occ[i, j]
        return shear


def get_num_slack_vars(aircraft: AircraftData, num_containers: int) -> dict:
    return {
        'pl_o': aircraft.num_positions,
        'pl_d': num_containers,
        'pl_w': math.floor(math.log2(aircraft.max_payload)) + 1
    }