import numpy as np
from dimod import as_bqm
from dwave.samplers import SimulatedAnnealingSampler

from aircraft_data import AircraftData
from coefficient_tuning import DataGenerator, tune_coef
from loading_problem import LoadingProblem
from raw_functions import no_overlaps_penalty, objective_f, maximum_capacity_penalty, no_duplicates_penalty, \
    contiguity_penalty
from utils import ResultThread

acft = AircraftData(2, 40, 40000, 26000, -0.1, 0.2)
slack_variables_counts = {'pl_o': 12, 'pl_d': 16, 'pl_w': 20}

# Tuning
coefs = {}
data_gen = DataGenerator(acft, (6, 6),
                         (650, 4000),
                         40000,
                         (120000, 120000),
                         slack_variables_counts)
pl_o_job = ResultThread(target=tune_coef, args=(data_gen, 5000, 0.0001, objective_f, no_overlaps_penalty,))
pl_w_job = ResultThread(target=tune_coef, args=(data_gen, 5000, 0.0000000001, objective_f, maximum_capacity_penalty,))
pl_d_job = ResultThread(target=tune_coef, args=(data_gen, 5000, 0.001, objective_f, no_duplicates_penalty,))
pl_c_job = ResultThread(target=tune_coef, args=(data_gen, 5000, 100, objective_f, contiguity_penalty,))
pl_o_job.start()
pl_w_job.start()
pl_d_job.start()
pl_c_job.start()
coefs['pl_o'] = pl_o_job.get_result()
coefs['pl_w'] = pl_w_job.get_result()
coefs['pl_d'] = pl_d_job.get_result()
coefs['pl_c'] = pl_c_job.get_result()
# coefs = {'pl_o': 1, 'pl_w': 1, 'pl_d': 1, 'pl_c': 1}
# Solve
cont_types = np.array(['t3', 't3', 't3', 't3', 't3', 't3'])
cont_masses = np.array([2134, 3455, 1866, 1699, 3500, 3332])
problem = LoadingProblem(acft, cont_types, cont_masses, 0.1, 120000, slack_variables_counts, coefs)
bqm = as_bqm(problem.get_q(), 'BINARY')

sampler = SimulatedAnnealingSampler()
result = sampler.sample(bqm, num_reads=1000).aggregate()

print(result)

cont_occ_solutions = problem.parse_solution(result)
cont_occ_solutions = problem.filter_solutions(cont_occ_solutions)
print(cont_occ_solutions)
