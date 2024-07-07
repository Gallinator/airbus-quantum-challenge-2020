import math
import numpy as np
from dimod import SampleSet, BQM, Binary

from aircraft_data import AircraftData
from utils import get_container_t, get_container_d, get_num_bits


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
        self.target_cg = target_cg * aircraft.payload_area_length
        self.zero_payload_mass = zero_payload_mass
        self.zero_payload_cg = zero_payload_cg * aircraft.payload_area_length
        self.container_t = get_container_t(container_types)
        self.container_d = get_container_d(container_types)
        self.coefficients = {'pl_o': 1.0,
                             'pl_w': 1.0,
                             'pl_d': 1.0,
                             'pl_c': 1.0,
                             'cl_t': 1.0,
                             'cl_u': 1.0,
                             'cl_l': 1.0,
                             'sl_l': 1.0,
                             'sl_r': 1.0}
        self.num_slack_variables = self.get_num_slack_vars()

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

    def get_x_cg(self) -> tuple[BQM, BQM]:
        x_cg_up = BQM.empty('BINARY')
        x_cg_down = BQM.empty('BINARY')
        for i, t_i in enumerate(self.container_t):
            for pos in range(self.aircraft.num_positions):
                x_cg_up += Binary(f'p_{i}_{pos}', t_i * self.container_masses[i] * self.aircraft.locations[pos])
                x_cg_down += Binary(f'p_{i}_{pos}', t_i * self.container_masses[i])
        x_cg_up += self.zero_payload_mass * self.zero_payload_cg
        x_cg_down += self.zero_payload_mass
        return x_cg_up, x_cg_down

    def get_cg_target_bqm(self) -> BQM:
        x_cg_up, x_cg_down = self.get_x_cg()
        bqm = x_cg_up - x_cg_down * self.target_cg
        bqm = bqm ** 2
        bqm *= self.coefficients['cl_t']
        return bqm

    def get_cg_lower_bqm(self):
        x_cg_up, x_cg_down = self.get_x_cg()
        bqm = x_cg_up - x_cg_down * self.aircraft.min_cg
        for k in range(self.num_slack_variables['cl_l']):
            bqm += Binary(f'v_cl_l_{k}', -(2 ** k))
        bqm = bqm ** 2
        bqm *= self.coefficients['cl_l']
        return bqm

    def get_cg_upper_bqm(self):
        x_cg_up, x_cg_down = self.get_x_cg()
        bqm = x_cg_up - x_cg_down * self.aircraft.max_cg
        for k in range(self.num_slack_variables['cl_u']):
            bqm += Binary(f'v_cl_u_{k}', 2 ** k)
        bqm = bqm ** 2
        bqm *= self.coefficients['cl_u']
        return bqm

    def get_left_shear_at_pos_bqm(self, u: int):
        bqm = BQM('BINARY')
        for i, t_i in enumerate(self.container_t):
            for pos in range(u + 1):
                bqm += Binary(f'p_{i}_{pos}', t_i * self.container_masses[i])
        return bqm

    def get_right_shear_at_pos_bqm(self, u: int):
        bqm = BQM('BINARY')
        for i, t_i in enumerate(self.container_t):
            for pos in range(u + 1, self.aircraft.num_positions):
                bqm += Binary(f'p_{i}_{pos}', t_i * self.container_masses[i])
        return bqm

    def get_left_shear_bqm(self):
        bqm = BQM('BINARY')
        half_pos = int(self.aircraft.num_positions / 2.0)
        for u in range(half_pos):
            u_bqm = self.get_left_shear_at_pos_bqm(u)
            for k in range(self.num_slack_variables['sl'][u]):
                u_bqm += Binary(f'v_sl_l_{u}_{k}', 2 ** k)
            u_bqm += -self.aircraft.shear_curve[u]
            bqm += u_bqm ** 2

        if self.aircraft.num_positions % 2 != 0:
            zero_bqm = BQM('BINARY')
            zero_bqm += self.get_left_shear_at_pos_bqm(half_pos - 1)
            for i, t_i in enumerate(self.container_t):
                zero_bqm += Binary(f'p_{i}_{half_pos}', t_i * self.container_masses[i] / 2)
            for k in range(self.num_slack_variables['sl'][half_pos]):
                zero_bqm += Binary(f'v_sl_l_c_0', 2 ** k)
            zero_bqm += -self.aircraft.shear_curve[half_pos]
            bqm += zero_bqm ** 2

        return self.coefficients['sl_l'] * bqm

    def get_right_shear_bqm(self):
        bqm = BQM('BINARY')
        half_pos = int(self.aircraft.num_positions / 2.0)
        is_even_pos = self.aircraft.num_positions % 2 == 0
        u_min = half_pos - 1 if is_even_pos else half_pos
        limit_offset = 1 if is_even_pos else 2

        for u in range(u_min, self.aircraft.num_positions - 1):
            u_bqm = self.get_right_shear_at_pos_bqm(u)
            for k in range(self.num_slack_variables['sl'][u + limit_offset]):
                u_bqm += Binary(f'v_sl_r_{u}_{k}', 2 ** k)
            u_bqm += -self.aircraft.shear_curve[u + limit_offset]
            bqm += u_bqm ** 2

        if not is_even_pos:
            zero_bqm = BQM('BINARY')
            zero_bqm += self.get_right_shear_at_pos_bqm(half_pos)
            for i, t_i in enumerate(self.container_t):
                zero_bqm += Binary(f'p_{i}_{half_pos}', t_i * self.container_masses[i] / 2)
            for k in range(self.num_slack_variables['sl'][half_pos + 1]):
                zero_bqm += Binary(f'v_sl_r_c_0', 2 ** k)
            zero_bqm += -self.aircraft.shear_curve[half_pos + 1]
            bqm += zero_bqm ** 2

        return self.coefficients['sl_r'] * bqm

    def get_bqm(self) -> BQM:
        obj_q = self.get_objective_bqm()
        no_overlaps_q = self.get_no_overlaps_bqm()
        no_duplicates_q = self.get_no_duplicates_bqm()
        max_capacity_q = self.get_max_capacity_bqm()
        contiguity_q = self.get_contiguity_bqm()
        cg_target = self.get_cg_target_bqm()
        cg_lower = self.get_cg_lower_bqm()
        cg_upper = self.get_cg_upper_bqm()
        shear_l = self.get_left_shear_bqm()
        shear_r = self.get_right_shear_bqm()
        return obj_q + no_overlaps_q + no_duplicates_q + max_capacity_q + contiguity_q + cg_target + cg_lower + cg_upper + shear_l + shear_r

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
                    self.check_contiguity_constraint(s) and
                    self.check_cg_upper_bound_constraint(s) and
                    self.check_cg_lower_bound_constraint(s)):
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

    def check_cg_lower_bound_constraint(self, cont_occ: np.ndarray) -> bool:
        return self.get_cg(cont_occ) >= self.aircraft.min_cg

    def check_cg_upper_bound_constraint(self, cont_occ: np.ndarray) -> bool:
        return self.get_cg(cont_occ) <= self.aircraft.max_cg

    def check_shear_constraint(self, cont_occ: np.ndarray) -> bool:
        shear = np.concatenate(self.get_shear(cont_occ))[:-1]
        return np.all(np.less(shear, self.aircraft.shear_curve[:-1]))

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
        shear_l = []
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
            shear_r = [s + shear_r[0]] + shear_r
            shear_l += [shear_l[-1] + s]

        return np.array(shear_l), np.array(shear_r)

    def get_shear_at_left(self, pos, cont_occ: np.ndarray):
        shear = 0
        for i, t_i in enumerate(self.container_t):
            for j in range(pos+1):
                shear += t_i * self.container_masses[i] * cont_occ[i, j]
        return shear

    def get_shear_at_right(self, pos, cont_occ: np.ndarray):
        shear = 0
        for i, t_i in enumerate(self.container_t):
            for j in range(pos + 1, self.aircraft.num_positions):
                shear += t_i * self.container_masses[i] * cont_occ[i, j]
        return shear

    def get_num_slack_vars(self) -> dict:
        return {
            'pl_o': self.aircraft.num_positions,
            'pl_d': len(self.container_types),
            'pl_w': get_num_bits(self.aircraft.max_payload),
            'cl_l': get_num_bits(self.aircraft.min_cg),
            'cl_u': get_num_bits(self.aircraft.max_cg),
            'sl': [get_num_bits(s) for s in self.aircraft.shear_curve]
        }
