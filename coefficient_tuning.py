import math
import numpy as np
from dimod import BQM


def get_coef(objective_q: BQM, penalty_q: BQM) -> float:
    penalty_values = list(penalty_q.to_qubo()[0].values())

    if len(penalty_values) == 0:
        return 0

    penalty_mean = np.mean(np.abs(penalty_values))
    if penalty_mean == 0:
        return 0

    obj_mean = np.mean(np.abs(list(objective_q.to_qubo()[0].values())))

    return math.pow(10, math.log10(obj_mean) - math.log10(penalty_mean))
