import math
import numpy as np


def get_coef(objective_q: dict, penalty_q: dict) -> float:
    penalty_values = list(penalty_q.values())

    if len(penalty_values) == 0:
        return 0

    penalty_mean = np.mean(np.abs(penalty_values))
    if penalty_mean == 0:
        return 0

    obj_mean = np.mean(list(objective_q.values()))

    return math.pow(10, math.log10(obj_mean) - math.log10(penalty_mean))
