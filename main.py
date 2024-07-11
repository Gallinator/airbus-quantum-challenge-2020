import dwave.inspector
import numpy as np
from dimod import as_bqm
from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

from aircraft_data import AircraftData
from coefficient_tuning import get_coef
from loading_problem import LoadingProblem
from utils import get_linear_shear_curve
from visualization import plot_solution


def get_tuned_coefs(problem: LoadingProblem) -> dict:
    coefs = {}
    coefs['pl_o'] = get_coef(problem.get_objective_bqm(), problem.get_no_overlaps_bqm())
    coefs['pl_w'] = get_coef(problem.get_objective_bqm(), problem.get_max_capacity_bqm())
    coefs['pl_d'] = get_coef(problem.get_objective_bqm(), problem.get_no_duplicates_bqm())
    coefs['pl_c'] = get_coef(problem.get_objective_bqm(), problem.get_contiguity_bqm())
    coefs['cl_u'] = get_coef(problem.get_objective_bqm(), problem.get_cg_upper_bqm())
    coefs['cl_l'] = get_coef(problem.get_objective_bqm(), problem.get_cg_lower_bqm())
    coefs['cl_t'] = get_coef(problem.get_objective_bqm(), problem.get_cg_target_bqm())
    return coefs


def main():
    use_real_sampler = input('Use real sampler? [y/n]').lower() == 'y'
    shear_curve = get_linear_shear_curve(4, 26000)
    acft = AircraftData(4, 40, 8000, shear_curve, -0.1, 0.2)
    cont_types = np.array(['t1', 't1', 't1', 't1', 't1', 't1'])
    cont_masses = np.array([2134, 3455, 1866, 1699, 3500, 3332])

    # Solve
    problem = LoadingProblem(acft, cont_types, cont_masses, 0.1, 120000, -0.05)
    problem.coefficients = get_tuned_coefs(problem)

    bqm = problem.get_bqm()

    if use_real_sampler:
        with Client.from_config() as client:
            available_solvers = client.get_solvers()

        for i, solver in enumerate(available_solvers):
            print(f'[{i}] - {solver.id}')
        solver = available_solvers[int(input('Select solver:'))]
        label = input("Problem label:")

        sampler = DWaveSampler(solver=solver.id)
        sampler_embedding = EmbeddingComposite(sampler)
        chain_str = max(np.abs(np.fromiter(bqm.to_qubo()[0].values(), dtype=float))) * 1.5
        result = sampler_embedding.sample(bqm, num_reads=1000, label=label, chain_strength=chain_str).aggregate()
        dwave.inspector.show(result)
    else:
        sampler = SimulatedAnnealingSampler()
        result = sampler.sample(bqm, num_reads=1000).aggregate()

    print(result)

    cont_occ_solutions = problem.parse_solution(result)
    cont_occ_solutions = problem.filter_solutions(cont_occ_solutions)

    print(f'{len(cont_occ_solutions)} feasible solutions found')

    plot_top_solutions(problem, cont_occ_solutions, 1)


def plot_top_solutions(problem: LoadingProblem, solutions_occ, k: int):
    payloads = [problem.get_payload_weight(o) for o in solutions_occ]
    indices = np.flip(np.argsort(payloads))
    indices = indices[:k]
    for i, sol_i in enumerate(indices):
        path = f'out/solution_{i + 1}.png'
        plot_solution(f'Solution payload {int(payloads[sol_i])} kg', problem, solutions_occ[sol_i], path)


if __name__ == '__main__':
    main()
