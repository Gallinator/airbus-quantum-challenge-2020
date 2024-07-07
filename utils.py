import threading
import traceback

import numpy as np


def get_linear_shear_curve(num_pos: int, limit: int):
    num_steps = num_pos // 2 + (1 if num_pos % 2 == 0 else 2)
    left = np.linspace(0, limit, num=num_steps)
    right = np.linspace(limit, 0, num=num_steps)
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
