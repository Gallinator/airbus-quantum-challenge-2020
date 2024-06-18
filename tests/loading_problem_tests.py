import unittest

import numpy as np
from dimod import BQM

from aircraft_data import get_default_aircraft, AircraftData
from loading_problem import LoadingProblem, get_squared_bqm


class LoadingProblemTests(unittest.TestCase):

    def test_container_mass_types_mismatch(self):
        acft = get_default_aircraft()
        cont_types = np.array(['t1', 't1'])
        cont_masses = np.array([100, 100, 100])
        with self.assertRaises(ValueError):
            LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

    def test_objective_q(self):
        acft = AircraftData(2, 10, 1, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_0_0', 'p_0_0'): -100,
                                ('p_0_1', 'p_0_1'): -100,
                                ('p_1_0', 'p_1_0'): -50,
                                ('p_1_1', 'p_1_1'): -50})

        self.assertEqual(problem.get_objective_bqm(), true_q)

    def test_no_overlaps_q(self):
        acft = AircraftData(2, 10, 1, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

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
        acft = AircraftData(2, 10, 1, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

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
        acft = AircraftData(2, 10, 1, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't3'])
        cont_masses = np.array([100, 100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = {('p_1_0', 'p_1_0'): 1 / 2,
                  ('p_1_1', 'p_1_1'): 1 / 2,
                  ('p_1_0', 'p_1_1'): -1,
                  ('p_2_0', 'p_2_0'): 1 / 2,
                  ('p_2_1', 'p_2_1'): 1 / 2,
                  ('p_2_0', 'p_2_1'): -1}

        self.assertEqual(problem.get_contiguity(), true_q)

    def test_max_capacity_q(self):
        acft = AircraftData(2, 10, 1, 0, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = {('p_0_0', 'p_0_0'): 80,
                  ('p_0_1', 'p_0_1'): 80,
                  ('p_1_0', 'p_1_0'): 15,
                  ('p_1_1', 'p_1_1'): 15,
                  ('v_w_0', 'v_w_0'): -1,
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

    def test_check_overlap_constraint_true(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0]])

        self.assertTrue(problem.check_overlap_constraint(cont_occ))

    def test_check_overlap_constraint_false(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_overlap_constraint(cont_occ))

    def test_check_contiguity_constraint_true(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_contiguity_constraint(cont_occ))

    def test_check_contiguity_constraint_false(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]])

        self.assertFalse(problem.check_contiguity_constraint(cont_occ))

    def test_check_max_weight_constraint_true(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_max_weight_constraint(cont_occ))

    def test_check_max_weight_constraint_false(self):
        acft = AircraftData(3, 10, 20, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_max_weight_constraint(cont_occ))

    def test_check_no_duplicates_constraint_true(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_no_duplicates_constraint(cont_occ))

    def test_check_no_duplicates_constraint_false(self):
        acft = AircraftData(3, 10, 20, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_no_duplicates_constraint(cont_occ))

    def test_get_payload_weight(self):
        acft = AircraftData(3, 10, 30, 0, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]])

        self.assertEqual(problem.get_payload_weight(cont_occ), 40)

    def test_get_squared_bqm(self):
        var_names = ['x1', 'x2']
        var_coefs = [2.0, -1.0]
        offset = -10.0

        true_qbm = BQM.from_qubo({('x1', 'x1'): -36.0,
                                  ('x2', 'x2'): 21.0,
                                  ('x1', 'x2'): -4.0}, 100.0)
        self.assertEqual(true_qbm, get_squared_bqm(var_names, var_coefs, offset))
