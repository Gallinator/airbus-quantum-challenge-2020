import numpy as np
from dimod import as_bqm
from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

from aircraft_data import AircraftData
from coefficient_tuning import DataGenerator, tune_coef
from loading_problem import LoadingProblem, get_num_slack_vars
from raw_functions import no_overlaps_penalty, objective_f, maximum_capacity_penalty, no_duplicates_penalty, \
    contiguity_penalty
from utils import ResultThread
from visualization import plot_solution


def get_tuned_coefs(data_gen: DataGenerator) -> dict:
    coefs = {}
    pl_o_job = ResultThread(target=tune_coef, args=(data_gen, 2000, objective_f, no_overlaps_penalty,))
    pl_w_job = ResultThread(target=tune_coef, args=(data_gen, 2000, objective_f, maximum_capacity_penalty,))
    pl_d_job = ResultThread(target=tune_coef, args=(data_gen, 2000, objective_f, no_duplicates_penalty,))
    pl_c_job = ResultThread(target=tune_coef, args=(data_gen, 2000, objective_f, contiguity_penalty,))
    pl_o_job.start()
    pl_w_job.start()
    pl_d_job.start()
    pl_c_job.start()
    coefs['pl_o'] = pl_o_job.get_result()
    coefs['pl_w'] = pl_w_job.get_result()
    coefs['pl_d'] = pl_d_job.get_result()
    coefs['pl_c'] = pl_c_job.get_result()
    return coefs


def main():
    use_real_sampler = input('Use real sampler? [y/n]').lower() == 'y'
    acft = AircraftData(4, 40, 8000, 26000, -0.1, 0.2)
    cont_types = np.array(['t1', 't1', 't1', 't1', 't1', 't1'])
    cont_masses = np.array([2134, 3455, 1866, 1699, 3500, 3332])

    # Tuning
    coefs = get_tuned_coefs(DataGenerator(acft, cont_masses, cont_types, get_num_slack_vars(acft, len(cont_types))))

    # Solve
    problem = LoadingProblem(acft, cont_types, cont_masses, 0.1, 120000, -0.05, coefs)
    bqm = as_bqm(problem.get_q(), 'BINARY')

    if use_real_sampler:
        with Client.from_config() as client:
            available_solvers = client.get_solvers()

        for i, solver in enumerate(available_solvers):
            print(f'[{i}] - {solver.id}')
        solver = available_solvers[int(input('Select solver:'))]
        label = input("Problem label:")

        sampler = DWaveSampler(solver=solver.id)
        sampler_embedding = EmbeddingComposite(sampler)
        result = sampler_embedding.sample(bqm, num_reads=100, label=label).aggregate()
    else:
        sampler = SimulatedAnnealingSampler()
        result = sampler.sample(bqm, num_reads=1000).aggregate()

    print(result)

    cont_occ_solutions = problem.parse_solution(result)
    cont_occ_solutions = problem.filter_solutions(cont_occ_solutions)

    plot_top_solutions(problem, cont_occ_solutions, 1)


def plot_top_solutions(problem: LoadingProblem, solutions_occ, k: int):
    payloads = [problem.get_payload_weight(o) for o in solutions_occ]
    indices = np.flip(np.argsort(payloads))
    indices = indices[:k]
    for i, sol_i in enumerate(indices):
        path = f'out/solution_{i + 1}.png'
        plot_solution(f'Solution payload {int(payloads[i])} kg', problem, solutions_occ[sol_i], path)


if __name__ == '__main__':
    main()
