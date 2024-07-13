import math
import numpy as np
from dimod import BQM
from dwave.samplers import SteepestDescentSampler


def get_coef(objective_q: BQM, penalty_q: BQM) -> float:
    penalty_values = list(penalty_q.to_qubo()[0].values())

    if len(penalty_values) == 0:
        return 0

    penalty_mean = np.mean(np.abs(penalty_values))
    if penalty_mean == 0:
        return 0

    obj_mean = np.mean(np.abs(list(objective_q.to_qubo()[0].values())))

    return math.pow(10, math.log10(obj_mean) - math.log10(penalty_mean))


def get_oom(value: float) -> int:
    if value == 0:
        return 0
    return math.floor(math.log10(abs(value)))


def tune_coef(obj_bqm: BQM, penalty_bqm: BQM, step_size, verbose=False):
    print(f'Tuning {penalty_bqm.shape}')
    sampler = SteepestDescentSampler()
    prev_state = None
    coef = step_size
    num_zero_results = 0
    while True:
        if num_zero_results == 5:
            return 0
        b1 = obj_bqm.copy(True)
        b2 = coef * penalty_bqm.copy(True)
        result = sampler.sample(b1 + b2, initial_states=prev_state)
        prev_state = result.aggregate().samples(1)[0]
        b1_val = b1.energy(prev_state)
        b2_val = b2.energy(prev_state)

        if verbose:
            print(f'Objective energy: {b1_val} Penalty energy: {b2_val}')
        if b2_val == 0:
            num_zero_results += 1
            continue
        if get_oom(b1_val) <= get_oom(b2_val):
            break
        coef += step_size
    print(f'Calculated coefficient {coef}')
    return coef
