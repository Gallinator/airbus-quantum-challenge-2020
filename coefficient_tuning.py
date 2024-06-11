import math
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
                 container_masses: np.ndarray,
                 container_types: np.ndarray,
                 num_slack_variables: dict):
        self.aircraft = aircraft
        self.cont_num = len(container_types)
        self.cont_masses = container_masses
        self.num_slack = num_slack_variables
        self.cont_types = container_types
        self.cont_t = get_container_t(container_types)
        self.cont_d = get_container_d(container_types)

    def generate(self, num_samples) -> np.array:
        data = np.empty(shape=num_samples, dtype=dict)
        for n in range(num_samples):
            d = {}
            d['cont_occ'] = generate_random_sparse_occupancies(self.cont_num, self.aircraft.num_positions)
            d['cont_types'] = self.cont_types
            d['cont_masses'] = self.cont_masses
            d['cont_d'] = self.cont_d
            d['cont_t'] = self.cont_t
            d['max_pl'] = self.aircraft.max_payload

            slacks = {}
            for v_k in self.num_slack.keys():
                slacks[v_k] = np.random.randint(low=0, high=2, size=self.num_slack[v_k])
            d['slack_vars'] = slacks
            data[n] = d

        return data


def tune_coef(data_generator: DataGenerator,
              num_samples: float,
              objective_f,
              penalty_f) -> float:
    penalty_outs = []
    obj_outs = []

    for s in data_generator.generate(num_samples):
        obj_outs.append(objective_f(**s))
        penalty_outs.append(penalty_f(**s))

    obj_mean = np.mean(obj_outs)
    obj_oom = get_oom(abs(obj_mean))
    penalty_mean = np.mean(penalty_outs)
    penalty_oom = get_oom(abs(penalty_mean))
    if penalty_mean == 0:
        coef = 0
    else:
        coef = math.pow(10, obj_oom - penalty_oom)

    print(f'{penalty_f.__name__}\n Penalty mean: {penalty_mean}, Objective mean: {obj_mean}, Coefficient: {coef}\n')
    return coef
