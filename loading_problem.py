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

    def get_no_overlaps_q(self) -> dict:
        num_slack = self.num_slack_variables['pl_o'] // self.aircraft.num_positions
        q = {}
        # Calculate q for all positions
        for pos in range(self.aircraft.num_positions):
            q_pos = {}
            # Quadratic terms with minus one
            for i, _ in enumerate(self.container_d):
                q_pos[(f'p_{i}_{pos}', f'p_{i}_{pos}')] = (self.container_d[i] ** 2) - 2 * self.container_d[i]

            for i in range(num_slack):
                q_pos[(f'v_o_{pos}_{i}', f'v_o_{pos}_{i}')] = (2 ** (2 * i)) - 2 * (2 ** i)

            # Right element fixed, add all combinations of p_i_j variables
            for i, _ in enumerate(self.container_d):
                for k in range(i):
                    q_pos[(f'p_{k}_{pos}', f'p_{i}_{pos}')] = 2 * self.container_d[i] * self.container_d[k]

            # Slack variables
            for i in range(num_slack):
                for k in range(i):
                    q_pos[(f'v_o_{pos}_{k}', f'v_o_{pos}_{i}')] = (2 ** (i + k + 1))
                for k, _ in enumerate(self.container_d):
                    q_pos[(f'p_{k}_{pos}', f'v_o_{pos}_{i}')] = 2 * self.container_d[k] * (2 ** i)

            q = merge_q([q, q_pos])

        q = adjust_with_coef(q, self.coefficients['pl_o'])
        return q

    def get_no_duplicates(self) -> dict:
        num_slack = self.num_slack_variables['pl_d'] // len(self.container_types)
        q = {}

        for c, t_i in enumerate(self.container_t):
            q_c = {}
            # Quadratic terms
            for pos in range(self.aircraft.num_positions):
                q_c[(f'p_{c}_{pos}', f'p_{c}_{pos}')] = (t_i ** 2) - 2 * t_i
            for k in range(num_slack):
                q_c[(f'v_d_{c}_{k}', f'v_d_{c}_{k}')] = (2 ** (2 * k)) - 2 * (2 ** k)

            # Right element fixed, add all combinations of p_i_j variables
            for pos in range(self.aircraft.num_positions):
                for k in range(pos):
                    q_c[(f'p_{c}_{k}', f'p_{c}_{pos}')] = 2 * t_i * t_i
            for k in range(num_slack):
                for j in range(self.aircraft.num_positions):
                    q_c[(f'p_{c}_{j}', f'v_d_{c}_{k}')] = 2 * (2 ** k) * t_i
                for j in range(k):
                    q_c[(f'v_d_{c}_{j}', f'v_d_{c}_{k}')] = (2 ** (j + k + 1))
            q = merge_q([q, q_c])
        q = adjust_with_coef(q, self.coefficients['pl_d'])
        return q

    def get_contiguity(self) -> dict:
        q = {}
        for i, t_i in enumerate(self.container_t):
            q_c = {}
            # Not a type 3 container
            if t_i >= 1:
                continue
            else:
                for pos in range(self.aircraft.num_positions - 1):
                    q_c[(f'p_{i}_{pos}', f'p_{i}_{pos}')] = 1 / 2
                    q_c[(f'p_{i}_{pos}', f'p_{i}_{pos + 1}')] = -1
                q[(f'p_{i}_{self.aircraft.num_positions - 1}', f'p_{i}_{self.aircraft.num_positions - 1}')] = 1 / 2
                q = merge_q([q, q_c])
        q = adjust_with_coef(q, self.coefficients['pl_c'])
        return q

    def get_max_capacity_q(self) -> dict:
        num_slack = self.num_slack_variables['pl_w']
        q = {}
        for i, t_i in enumerate(self.container_t):
            for pos in range(self.aircraft.num_positions):
                coef = t_i * self.container_masses[i]
                q[f'p_{i}_{pos}', f'p_{i}_{pos}'] = (coef ** 2) - 2 * self.aircraft.max_payload * coef
        for k in range(num_slack):
            q[(f'v_w_{k}', f'v_w_{k}')] = (2 ** (2 * k)) - (2 ** (k + 1)) * self.aircraft.max_payload

        # p_i_j by expanding the double sum
        num_cont = len(self.container_t)
        num_pos = self.aircraft.num_positions
        for i_expanded in range(num_pos * num_cont):
            i = i_expanded // num_pos
            pos_i = i_expanded - num_pos * i
            for k_expanded in range(i_expanded):
                k = k_expanded // num_pos
                pos_k = k_expanded - num_pos * k
                q[f'p_{k}_{pos_k}', f'p_{i}_{pos_i}'] = 2 * (self.container_t[i] * self.container_masses[i] *
                                                             self.container_t[k] * self.container_masses[k])

        for k in range(num_slack):
            for j in range(k):
                q[f'v_w_{j}', f'v_w_{k}'] = (2 ** (k + j + 1))
            for i_expanded in range(num_pos * num_cont):
                i = i_expanded // num_pos
                pos = i_expanded - i * num_pos
                q[f'p_{i}_{pos}', f'v_w_{k}'] = self.container_t[i] * self.container_masses[i] * (2 ** (k + 1))

        q = adjust_with_coef(q, self.coefficients['pl_w'])
        return q

    def get_q(self) -> dict:
        obj_q = self.get_objective_bqm()
        no_overlaps_q = self.get_no_overlaps_q()
        no_duplicates_q = self.get_no_duplicates()
        max_capacity_q = self.get_max_capacity_q()
        contiguity_q = self.get_contiguity()
        return merge_q([obj_q, no_overlaps_q, no_duplicates_q, max_capacity_q, contiguity_q])

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


def q_reducer(accumulator, element):
    for key, value in element.items():
        s = accumulator.get(key, 0) + value
        if s != 0:
            accumulator[key] = s
    return accumulator


def merge_q(q: list):
    return reduce(q_reducer, q, {})


def adjust_with_coef(q: dict, coef: float) -> dict:
    res = q.copy()
    for k in q.keys():
        res[k] *= coef
    return res


def get_squared_bqm(var_names: list[str], var_coefs: list[float], offset: float) -> BQM:
    variables = Binaries(var_names)
    sum_bqm = BQM.empty('BINARY')
    for var, coef in zip(variables, var_coefs):
        sum_bqm = sum_bqm + coef * var
    return (sum_bqm + offset) ** 2
