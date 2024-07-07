import unittest

import numpy as np
from dimod import BQM

from aircraft_data import get_default_aircraft, AircraftData
from loading_problem import LoadingProblem
from utils import get_linear_shear_curve


class LoadingProblemTests(unittest.TestCase):

    def test_container_mass_types_mismatch(self):
        acft = get_default_aircraft()
        cont_types = np.array(['t1', 't1'])
        cont_masses = np.array([100, 100, 100])
        with self.assertRaises(ValueError):
            LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

    def test_objective_q(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 10, 1, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_0_0', 'p_0_0'): -100,
                                ('p_0_1', 'p_0_1'): -100,
                                ('p_1_0', 'p_1_0'): -50,
                                ('p_1_1', 'p_1_1'): -50})

        self.assertEqual(problem.get_objective_bqm(), true_q)

    def test_no_overlaps_q(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 10, 1, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_0_0', 'p_1_0'): 2,
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
                                ('v_o_1_0', 'v_o_1_0'): -1}, 2)

        self.assertEqual(problem.get_no_overlaps_bqm(), true_q)

    def test_no_duplicates_q(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 10, 1, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_0_0', 'p_0_1'): 2,
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
                                ('v_d_1_0', 'v_d_1_0'): -1}, 2.0)

        self.assertEqual(problem.get_no_duplicates_bqm(), true_q)

    def test_contiguity_q(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 10, 1, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't3'])
        cont_masses = np.array([100, 100, 100])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_1_0', 'p_1_0'): 1 / 2,
                                ('p_1_1', 'p_1_1'): 1 / 2,
                                ('p_1_0', 'p_1_1'): -1,
                                ('p_2_0', 'p_2_0'): 1 / 2,
                                ('p_2_1', 'p_2_1'): 1 / 2,
                                ('p_2_0', 'p_2_1'): -1})

        self.assertEqual(problem.get_contiguity_bqm(), true_q)

    def test_max_capacity_q(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 10, 1, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3'])
        cont_masses = np.array([10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        true_q = BQM.from_qubo({('p_0_0', 'p_0_0'): 80,
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
                                ('p_1_1', 'v_w_0'): 10}, 1)

        self.assertEqual(problem.get_max_capacity_bqm(), true_q)

    def test_cg_target_bqm(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 8, 1, shear_curve, -1, 1)
        cont_types = np.array(['t1'])
        cont_masses = np.array([10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.125, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_0', 'p_0_0'): 2220,
                                  ('p_0_1', 'p_0_1'): -340,
                                  ('p_0_0', 'p_0_1'): -600}, 484)

        self.assertEqual(problem.get_cg_target_bqm(), true_bqm)

    def test_cg_lower_bqm(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_0', 'p_0_0'): -260,
                                  ('p_0_1', 'p_0_1'): 1980,
                                  ('p_0_0', 'p_0_1'): -600,
                                  ('v_cl_l_0', 'v_cl_l_0'): -35,
                                  ('p_0_0', 'v_cl_l_0'): 20,
                                  ('p_0_1', 'v_cl_l_0'): -60}, 324)

        self.assertEqual(problem.get_cg_lower_bqm(), true_bqm)

    def test_cg_upper_bqm(self):
        shear_curve = get_linear_shear_curve(2, 26000)
        acft = AircraftData(2, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_0', 'p_0_0'): 2220,
                                  ('p_0_1', 'p_0_1'): -340,
                                  ('p_0_0', 'p_0_1'): -600,
                                  ('v_cl_u_0', 'v_cl_u_0'): -43,
                                  ('p_0_1', 'v_cl_u_0'): 20,
                                  ('p_0_0', 'v_cl_u_0'): -60}, 484)

        self.assertEqual(problem.get_cg_upper_bqm(), true_bqm)

    def test_shear_left_even(self):
        shear_curve = get_linear_shear_curve(2, 1)
        acft = AircraftData(2, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([1])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_0', 'p_0_0'): -1,
                                  ('v_sl_l_0_0', 'v_sl_l_0_0'): -1,
                                  ('p_0_0', 'v_sl_l_0_0'): 2}, 1)

        self.assertEqual(problem.get_left_shear_bqm(), true_bqm)

    def test_shear_right_even(self):
        shear_curve = get_linear_shear_curve(2, 1)
        acft = AircraftData(2, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([1])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_1', 'p_0_1'): -1,
                                  ('v_sl_r_0_0', 'v_sl_r_0_0'): -1,
                                  ('p_0_1', 'v_sl_r_0_0'): 2}, 1)

        self.assertEqual(problem.get_right_shear_bqm(), true_bqm)

    def test_shear_left_odd(self):
        shear_curve = get_linear_shear_curve(3, 1)
        acft = AircraftData(3, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([1])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)
        true_bqm = BQM.from_qubo({('p_0_0', 'p_0_0'): -1,
                                  ('v_sl_l_0_0', 'v_sl_l_0_0'): 0,
                                  ('p_0_0', 'p_0_0'): -1,
                                  ('p_0_1', 'p_0_1'): -3 / 4,
                                  ('v_sl_l_c_0', 'v_sl_l_c_0'): -1,
                                  ('p_0_0', 'p_0_1'): 1,
                                  ('p_0_0', 'v_sl_l_c_0'): 2,
                                  ('p_0_1', 'v_sl_l_c_0'): 1,
                                  ('p_0_0', 'v_sl_l_0_0'): 2}, 1.25)

        self.assertEqual(problem.get_left_shear_bqm(), true_bqm)

    def test_shear_right_odd(self):
        shear_curve = get_linear_shear_curve(3, 1)
        acft = AircraftData(3, 8, 100, shear_curve, -0.125, 0.125)
        cont_types = np.array(['t1'])
        cont_masses = np.array([1])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0625, 20, -0.0125)

        true_bqm = BQM.from_qubo({('p_0_1', 'p_0_1'): -3 / 4,
                                  ('p_0_2', 'p_0_2'): -1,
                                  ('v_sl_r_c_0', 'v_sl_r_c_0'): -1,
                                  ('v_sl_r_1_0', 'v_sl_r_1_0'): 0,
                                  ('p_0_1', 'p_0_2'): 1,
                                  ('p_0_1', 'v_sl_r_c_0'): 1,
                                  ('p_0_2', 'v_sl_r_c_0'): 2,
                                  ('p_0_2', 'v_sl_r_1_0'): 2}, 1.25)
        test_bqm = problem.get_right_shear_bqm()

        self.assertEqual(test_bqm, true_bqm)

    def test_check_overlap_constraint_true(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0]])

        self.assertTrue(problem.check_overlap_constraint(cont_occ))

    def test_check_overlap_constraint_false(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_overlap_constraint(cont_occ))

    def test_check_contiguity_constraint_true(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_contiguity_constraint(cont_occ))

    def test_check_contiguity_constraint_false(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]])

        self.assertFalse(problem.check_contiguity_constraint(cont_occ))

    def test_check_max_weight_constraint_true(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_max_weight_constraint(cont_occ))

    def test_check_max_weight_constraint_false(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 20, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_max_weight_constraint(cont_occ))

    def test_check_no_duplicates_constraint_true(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]])

        self.assertTrue(problem.check_no_duplicates_constraint(cont_occ))

    def test_check_no_duplicates_constraint_false(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 20, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0]])

        self.assertFalse(problem.check_no_duplicates_constraint(cont_occ))

    def test_get_payload_weight(self):
        shear_curve = get_linear_shear_curve(3, 26000)
        acft = AircraftData(3, 10, 30, shear_curve, 0, 0)
        cont_types = np.array(['t1', 't3', 't2'])
        cont_masses = np.array([10, 10, 10])
        problem = LoadingProblem(acft, cont_types, cont_masses, 0.0, 120000, -0.05)

        cont_occ = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]])

        self.assertEqual(problem.get_payload_weight(cont_occ), 40)
