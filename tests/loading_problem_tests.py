import unittest

import numpy as np

from aircraft_data import get_default_aircraft, AircraftData
from loading_problem import LoadingProblem


class LoadingProblemTests(unittest.TestCase):

    def test_container_mass_types_mismatch(self):
        acft = get_default_aircraft()
        cont_types = np.array(['t1', 't1'])
        cont_masses = np.array([100, 100, 100])
        with self.assertRaises(ValueError):
            LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {}, {})

    def test_objective_q(self):
        acft = AircraftData(2, 10, 0, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {}, {})

        true_q = {('p_0_0', 'p_0_0'): -100,
                  ('p_0_1', 'p_0_1'): -100,
                  ('p_1_0', 'p_1_0'): -50,
                  ('p_1_1', 'p_1_1'): -50}

        self.assertEqual(problem.get_objective_q(), true_q)

    def test_no_overlaps_q(self):
        acft = AircraftData(2, 10, 0, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {'pl_o': 2}, {'pl_o': 1})

        true_q = {('p_0_0', 'p_1_0'): 2,
                  ('p_0_0', 'v_o_0_0'): 2,
                  ('p_1_0', 'v_o_0_0'): 2,
                  ('p_0_0', 'p_0_0'): -1,
                  ('p_1_0', 'p_1_0'): -1,
                  ('v_o_0_0', 'v_o_0_0'): -1,
                  ('p_0_1', 'p_1_1'): 2,
                  ('p_0_1', 'v_o_1_0'): 2,
                  ('p_1_1', 'v_o_1_0'): 2,
                  ('p_0_1', 'p_0_1'): -1,
                  ('p_1_1', 'p_1_1'): -1,
                  ('v_o_1_0', 'v_o_1_0'): -1}

        self.assertEqual(problem.get_no_overlaps_q(), true_q)

    def test_no_duplicates_q(self):
        acft = AircraftData(2, 10, 0, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {'pl_d': 2}, {'pl_d': 1})

        true_q = {('p_0_0', 'p_0_1'): 2,
                  ('p_0_0', 'v_d_0_0'): 2,
                  ('p_0_1', 'v_d_0_0'): 2,
                  ('p_0_0', 'p_0_0'): -1,
                  ('p_0_1', 'p_0_1'): -1,
                  ('v_d_0_0', 'v_d_0_0'): -1,
                  ('p_1_0', 'p_1_1'): 1 / 2,
                  ('p_1_0', 'v_d_1_0'): 1,
                  ('p_1_1', 'v_d_1_0'): 1,
                  ('p_1_0', 'p_1_0'): -3 / 4,
                  ('p_1_1', 'p_1_1'): -3 / 4,
                  ('v_d_1_0', 'v_d_1_0'): -1}

        self.assertEqual(problem.get_no_duplicates(), true_q)

    def test_contiguity_q(self):
        acft = AircraftData(2, 10, 0, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't3'])
        cont_masses = np.array([100, 100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {}, {'pl_c': 1})

        true_q = {('p_1_0', 'p_1_0'): 1 / 2,
                  ('p_1_1', 'p_1_1'): 1 / 2,
                  ('p_1_0', 'p_1_1'): -1,
                  ('p_2_0', 'p_2_0'): 1 / 2,
                  ('p_2_1', 'p_2_1'): 1 / 2,
                  ('p_2_0', 'p_2_1'): -1}

        self.assertEqual(problem.get_contiguity(), true_q)

    def test_max_capacity_q(self):
        acft = AircraftData(2, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, {'pl_w': 1}, {'pl_w': 1})

        true_q = {('p_0_0', 'p_0_0'): -500,
                  ('p_0_1', 'p_0_1'): -500,
                  ('p_1_0', 'p_1_0'): -275,
                  ('p_1_1', 'p_1_1'): -275,
                  ('v_w_0', 'v_w_0'): -59,
                  ('p_0_0', 'p_0_1'): 200,
                  ('p_0_0', 'p_1_0'): 100,
                  ('p_0_1', 'p_1_0'): 100,
                  ('p_0_0', 'p_1_1'): 100,
                  ('p_0_1', 'p_1_1'): 100,
                  ('p_1_0', 'p_1_1'): 50,
                  ('p_0_0', 'v_w_0'): 20,
                  ('p_0_1', 'v_w_0'): 20,
                  ('p_1_0', 'v_w_0'): 10,
                  ('p_1_1', 'v_w_0'): 10}

        self.assertEqual(problem.get_max_capacity_q(), true_q)
