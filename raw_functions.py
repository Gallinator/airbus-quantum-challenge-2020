import numpy as np


def objective_f(cont_t: np.ndarray, cont_masses: np.ndarray, cont_occ: np.ndarray, **_):
    num_cont = cont_occ.shape[0]
    num_pos = cont_occ.shape[1]
    o_sum = 0
    for i in range(num_cont):
        for j in range(num_pos):
            o_sum += cont_t[i] * cont_masses[i] * cont_occ[i, j]
    return -o_sum


def no_overlaps_penalty(cont_occ: np.ndarray, cont_d: np.ndarray, slack_vars: dict, **_) -> float:
    out = 0
    num_pos = cont_occ.shape[1]
    for j in range(num_pos):
        out_j = 0
        for i, d_i in enumerate(cont_d):
            out_j += d_i * cont_occ[i, j]
        for k, v_k in enumerate(slack_vars['pl_o']):
            out_j += (2 ** k) * v_k
        out_j -= 1
        out += out_j**2
    return out


def no_duplicates_penalty(cont_occ: np.ndarray, cont_t: np.ndarray, slack_vars: dict, **_) -> float:
    out = 0
    num_cont = cont_occ.shape[0]
    num_pos = cont_occ.shape[1]
    for i in range(num_cont):
        out_i = 0
        for pos in range(num_pos):
            out += cont_t[i] * cont_occ[i, pos]
        for k, v_k in enumerate(slack_vars['pl_d']):
            out += (2 ** k) * v_k
        out_i -= 1
        out_i = out_i ** 2
        out += out_i
    return out


def contiguity_penalty(cont_occ: np.ndarray, cont_types: np.ndarray, **__) -> float:
    num_pos = cont_occ.shape[1]
    out = 0
    for i, c_type in enumerate(cont_types):
        if c_type != 't3':
            continue
        for pos in range(num_pos - 1):
            out += 1 / 2 * cont_occ[i, pos] - cont_occ[i, pos] * cont_occ[i, pos + 1]
        out += 1 / 2 * cont_occ[i, -1]
    return out


def maximum_capacity_penalty(cont_occ: np.ndarray,
                             cont_t: np.ndarray,
                             cont_masses: np.ndarray,
                             slack_vars: dict,
                             max_pl, **_) -> float:
    out = 0
    for index, p in np.ndenumerate(cont_occ):
        i, _ = index
        out += cont_t[i] * cont_masses[i] * p
    for k, v_k in enumerate(slack_vars['pl_w']):
        out += (2 ** k) * v_k
    out -= max_pl
    return out ** 2
