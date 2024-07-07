# Airbus Quantum Computing Challenge 2020

Implementation of the solution to the Airbus Quantum Challenge 2020 aircraft loading problem proposed
in [Aircraft Loading Optimization -- QUBO models under multiple constraints](https://arxiv.org/abs/2102.09621).<br>
The optimization process can be run on both a simulated annealer and real D-Wave solver.<br>
The current implementation lacks an effective strategy to tune the coefficients so a lot of runs are required.

## How to run

- Install requirements ``` pip install -r requirements.txt ```
- [Optional] Create an account at [D-Wave Leap](https://cloud.dwavesys.com/leap/signup/)
- [Optional] Take your API key then run ``` dwave config create ```
- Run ``` python main.py ```

## Customize problem parameters

To change the aircraft data create an ``` AircraftData ``` object and a shear curve. Two functions are provided to create symmetric and asymmetric linear curves.<br> 

``` python
    shear_curve = get_linear_shear_curve(4, 26000)
    acft = AircraftData(num_positions=4, payloa_area_length=40, max_payload=8000, shear_curve=shear_curve, min_cg=-0.1, max_cg=0.2)
```

To change the problem requirements first define the container types and masses as lists.<br>
Each point of the shear curve must correspond to a container position with the center elements representing the central position left and right shears.

``` python
    cont_types = np.array(['t1', 't1', 't1', 't1', 't1', 't1'])
    cont_masses = np.array([2134, 3455, 1866, 1699, 3500, 3332])
```

Define the penalty functions coefficients as a ``` dict ```<br>

```python
    coefs = {'pl_o': 1.0,
             'pl_w': 1.0,
             'pl_d': 1.0,
             'pl_c': 1.0,
             'cl_u': 1.0,
             'cl_l': 1.0,
             'cl_t': 1.0,
             'sl_l': 1.0,
             'sl_r': 1.0}
```

Finally create a new ``` LoadingProblem ``` and set its coefficients<br>

```python
    problem = LoadingProblem(acft, cont_types, cont_masses, 0.1, 120000, -0.05)
    problem.coefficients = coefs
```

The ``` LoadingProblem ``` class allows to get the BQM representation of each constraint as well as the total problem to
pass to the solver. For example to create a BQM to pass to the solvers use<br>

```python
    bqm = problem.get_bqm()
```

The plot functions assume feasible solutions so first parse the solver results then remove the unfeasible solutoins. The
plot the top k solutions.<br>

```python
    cont_occ_solutions = problem.parse_solution(result)
    cont_occ_solutions = problem.filter_solutions(cont_occ_solutions)

    plot_top_solutions(problem, cont_occ_solutions, 1)
```

Plots will be saved in the ``` out ``` directory.<br><br>
![alt text](https://github.com/Gallinator/airbus-quantum-challenge-2020/blob/master/docs/example_plot.png)

## Todo

- [x] Payload constraints
- [x] CG constraints
- [x] Shear constraints
- [ ] Improve coefficient tuning. This should increase the ratio of feasible solutions found by the solvers
- [ ] Try hybrid solvers
