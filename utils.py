import math
import threading
import traceback

import numpy as np


def get_num_bits(value) -> int:
    """
    Computes the number of bits required to represent a value.
    If -1 < v < 1 returns 1 as the value has to be represented by some bit.
    :param value: The number to represent
    :return: The number of bits required to represent abs(v)
    """
    if -1 < value < 1:
        return 1
    return 1 if value == 0 else abs(int(math.log2(abs(value)))) + 1


def get_linear_left_curve(num_pos, limit):
    if num_pos % 2 == 0:
        num_steps = num_pos // 2 + 1
        step_size = limit / (num_steps - 1)
    else:
        num_steps = num_pos // 2 + 2
        step_size = limit / (num_steps - 1.5)
    left = [i * step_size for i in range(num_steps - 1)]
    return np.append(left, [limit])


def get_linear_shear_curve(num_pos: int, limit: int):
    left = get_linear_left_curve(num_pos, limit)
    right = np.flip(left)
    return np.concatenate((left[1:], right))


def get_linear_asym_shear_curve(num_pos: int, limit_l: int, limit_r: int):
    left = get_linear_left_curve(num_pos, limit_l)
    right = np.flip(get_linear_left_curve(num_pos, limit_r))
    return np.concatenate((left[1:], right))


def get_container_t(container_types: np.ndarray):
    t = np.ones(len(container_types))
    t[container_types == 't3'] = 1 / 2
    return t


def get_container_d(container_types: np.ndarray):
    d = np.ones(len(container_types))
    d[container_types == 't2'] = 1 / 2
    return d


class ResultThread(threading.Thread):
    """
    Simple class that wraps a thread allowing to get the result of the function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as e:
            traceback.print_exc()
            self.result = e

    def get_result(self, timeout: float | None = None):
        """
        This is a blocking operation
        :param timeout:
        :return: The result of the thread or an exception if the execution fails
        """
        self.join(timeout)
        return self.result
