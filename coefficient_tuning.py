import math
import multiprocessing

import numpy as np
from dimod import BQM
from dwave.samplers import SteepestDescentSampler

from loading_problem import LoadingProblem


def tune_coefs_average(problem: LoadingProblem) -> dict:
    coefs = {}
    coefs['pl_o'] = calc_coef_average(problem.get_objective_bqm(), problem.get_no_overlaps_bqm())
    coefs['pl_w'] = calc_coef_average(problem.get_objective_bqm(), problem.get_max_capacity_bqm())
    coefs['pl_d'] = calc_coef_average(problem.get_objective_bqm(), problem.get_no_duplicates_bqm())
    coefs['pl_c'] = calc_coef_average(problem.get_objective_bqm(), problem.get_contiguity_bqm())
    coefs['cl_u'] = calc_coef_average(problem.get_objective_bqm(), problem.get_cg_upper_bqm())
    coefs['cl_l'] = calc_coef_average(problem.get_objective_bqm(), problem.get_cg_lower_bqm())
    coefs['cl_t'] = calc_coef_average(problem.get_objective_bqm(), problem.get_cg_target_bqm())
    coefs['sl_l'] = calc_coef_average(problem.get_objective_bqm(), problem.get_left_shear_bqm())
    coefs['sl_r'] = calc_coef_average(problem.get_objective_bqm(), problem.get_right_shear_bqm())
    return coefs


def tune_task(coef_id, obj, penalty, step_size):
    print(f'Tuning {coef_id} coefficient')
    coef = coef_id, calc_coef_iterative(obj, penalty, step_size)
    print(f'{coef_id} coeffiecient {coef}')
    return coef


def tune_coefs_iterative(problem: LoadingProblem, step_sizes=None) -> dict:
    if step_sizes is None:
        step_sizes = {'pl_o': 1, 'pl_w': 0.01, 'pl_d': 0.1, 'pl_c': 1,
                      'cl_t': 0.0000000001, 'cl_u': 0.000000001, 'cl_l': 0.0000000001,
                      'sl_l': 0.1, 'sl_r': 0.01}
    with multiprocessing.Pool() as pool:
        args = [
            ('pl_o', problem.get_objective_bqm(), problem.get_no_overlaps_bqm(), step_sizes['pl_o']),
            ('pl_w', problem.get_objective_bqm(), problem.get_max_capacity_bqm(), step_sizes['pl_w']),
            ('pl_c', problem.get_objective_bqm(), problem.get_contiguity_bqm(), step_sizes['pl_c']),
            ('pl_d', problem.get_objective_bqm(), problem.get_no_duplicates_bqm(), step_sizes['pl_d']),
            ('cl_u', problem.get_objective_bqm(), problem.get_cg_upper_bqm(), step_sizes['cl_u']),
            ('cl_l', problem.get_objective_bqm(), problem.get_cg_lower_bqm(), step_sizes['cl_l']),
            ('cl_t', problem.get_objective_bqm(), problem.get_cg_target_bqm(), step_sizes['cl_t']),
            ('sl_l', problem.get_objective_bqm(), problem.get_left_shear_bqm(), step_sizes['sl_l']),
            ('sl_r', problem.get_objective_bqm(), problem.get_right_shear_bqm(), step_sizes['sl_r'])
        ]
        coefs = {}
        for coef_id, c in pool.starmap(tune_task, args):
            coefs[coef_id] = c
        return coefs


def calc_coef_average(objective_q: BQM, penalty_q: BQM) -> float:
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


def calc_coef_iterative(obj_bqm: BQM, penalty_bqm: BQM, step_size, verbose=False):
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
