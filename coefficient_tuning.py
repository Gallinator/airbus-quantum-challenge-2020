import math
import random

import numpy as np
import scipy

from aircraft_data import AircraftData
from utils import get_container_d, get_container_t


def generate_random_sparse_occupancies(n_cont, n_pos) -> np.ndarray:
    return scipy.sparse.random(n_cont, n_pos, density=0.5, format='csr', data_rvs=np.ones, dtype='f').toarray()


def get_oom(value):
    if value == 0:
        return 0
    return math.floor(math.log(value, 10))


class DataGenerator:
    def __init__(self, aircraft: AircraftData,
                 container_num_range: tuple,
                 container_mass_range: tuple,
                 max_payload: float,
                 zero_payload_mass_range: tuple,
                 num_slack_variables: dict):
        self.aircraft = aircraft
        self.cont_num_max = container_num_range[1]
        self.cont_num_min = container_num_range[0]
        self.cont_mass_max = container_mass_range[1]
        self.cont_mass_min = container_mass_range[0]
        self.zero_pl_max = zero_payload_mass_range[1]
        self.zero_pl_min = zero_payload_mass_range[0]
        self.max_pl = max_payload
        self.num_slack = num_slack_variables

    def generate(self, num_samples) -> np.ndarray:
        data = np.empty(shape=num_samples, dtype=dict)

        for n in range(num_samples):
            d = {}
            n_cont = random.randint(self.cont_num_min, self.cont_num_max)

            d['cont_occ'] = generate_random_sparse_occupancies(n_cont, self.aircraft.num_positions)

            cont_types = np.random.choice(['t1', 't2', 't3'], n_cont)
            d['cont_types'] = cont_types
            d['cont_d'] = get_container_d(cont_types)
            d['cont_t'] = get_container_t(cont_types)
            d['cont_masses'] = np.random.randint(low=self.cont_mass_min, high=self.cont_mass_max, size=n_cont)

            d['max_pl'] = self.max_pl

            slacks = {}
            for v_k in self.num_slack.keys():
                slacks[v_k] = np.random.randint(low=0, high=2, size=self.num_slack[v_k])
            d['slack_vars'] = slacks

            data[n] = d

        return data


def tune_coef(data_generator: DataGenerator,
              num_samples: int,
              step_increase: float,
              objective_f,
              penalty_f) -> float:
    penalty_oom = -1.0
    obj_oom = 0.0
    coef = 0.0

    while penalty_oom < obj_oom:
        coef += step_increase
        penalty_outs = []
        obj_outs = []
        data = data_generator.generate(num_samples)

        for _, s in enumerate(data):
            obj_outs.append(objective_f(**s))
            penalty_outs.append(coef * penalty_f(**s))

        obj_oom = get_oom(abs(np.mean(obj_outs)))
        penalty_oom = get_oom(abs(np.mean(penalty_outs)))

        print(f'{penalty_f.__name__}\n Penalty oom: {penalty_oom}, Objective oom: {obj_oom}, Coefficient: {coef}\n')

    print(f'Calculated coefficient: {coef}')
    return coef
