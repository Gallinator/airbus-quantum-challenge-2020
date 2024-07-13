import dwave.inspector
import numpy as np
from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

from aircraft_data import AircraftData
from coefficient_tuning import tune_coefs_average
from loading_problem import LoadingProblem
from utils import get_linear_shear_curve
from visualization import plot_solution


def main():
    use_real_sampler = input('Use real sampler? [y/n]').lower() == 'y'
    shear_curve = get_linear_shear_curve(4, 26000)
    acft = AircraftData(4, 40, 8000, shear_curve, -0.1, 0.2)
    cont_types = np.array(['t1', 't1', 't1', 't1', 't1', 't1'])
    cont_masses = np.array([2134, 3455, 1866, 1699, 3500, 3332])

    # Solve
    problem = LoadingProblem(acft, cont_types, cont_masses, 0.1, 120000, -0.05)
    problem.coefficients = tune_coefs_average(problem)
    bqm = problem.get_bqm()

    if use_real_sampler:
        with Client.from_config() as client:
            available_solvers = client.get_solvers(hybrid=False)

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
