"""
Solve a Continuous Problem with various Numerical Optimization Algorithms.

In this example, we try to solve the well-known Rosenbrock function. This
function is non-separable and has its optimum at the vector with all "1"
values. [1, 2]

We apply several numerical optimization methods that are included from
well-tested and established standard packages and wrapped into the `moptipy`
API. If you want to see the exactly same example, but with logging of the
progress of the algorithms enabled, check
`continuous_optimization_with_logging.py`.
If you want to look at the structured experiment execution API, you may want
to read `experiment_2_algorithms_4_problems.py`.

1. https://www.sfu.ca/%7Essurjano/rosen.html
2. http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files\
/TestGO_files/Page2537.htm
"""
from moptipy.algorithms.so.vector.cmaes_lib import (
    CMAES,  # the covariance matrix adaptation evolution strategy (CMA-ES)
    BiPopCMAES,  # the Bi-Population CMA-ES
    SepCMAES,  # the separable CMA-ES
)
from moptipy.algorithms.so.vector.pdfo import BOBYQA  # Powell's BOBYQA
from moptipy.algorithms.so.vector.scipy import (
    BGFS,  # Broyden/Fletcher/Goldfarb/Shanno from SciPy
    CG,  # conjugate gradient method from SciPy
    DE,  # differential evolution method from SciPy
    SLSQP,  # Sequential Least Squares Programming from SciPy
    TNC,  # Truncated Newton Method from SciPy
    NelderMead,  # Downhill Simplex from SciPy
    Powell,  # another algorithm by Powell from SciPy
)
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.vectorspace import VectorSpace


class Rosenbrock(Objective):
    """The Rosenbrock function with optimum [1, 1, ..., 1]."""

    def evaluate(self, x) -> float:
        """Compute the value of the Rosenbrock function."""
        return float((100.0 * sum((x[1:] - (x[:-1] ** 2)) ** 2))
                     + sum((x - 1.0) ** 2))

    def __str__(self) -> str:
        """Get the name of this problem."""
        return "rosenbrock"


# the setup of the problems and search space
space = VectorSpace(4, -10.0, 10.0)  # 4-D space with bounds [-10, 10]
f = Rosenbrock()  # the instance of the optimization problem
op0 = Op0Uniform(space)  # the nullary search operator: random uniform
b = space.create()  # a variable to store the best solution

# Perform one single run for a variety of different optimization algorithms.
for algorithm in [BGFS(op0, space),  # Broyden/Fletcher/Goldfarb/Shanno
                  BiPopCMAES(space),  # the bi-population CMA-ES
                  BOBYQA(op0, space),  # Bound Optimization by Quadrat. Apprx.
                  CG(op0, space),  # conjugate gradient method
                  CMAES(space),  # covariance matrix adaptation ES
                  DE(space),  # differential evolution
                  NelderMead(op0, space),  # downhill simplex
                  Powell(op0, space),  # other Powell method (besides BOBYQA)
                  SepCMAES(space),  # the separable CMA-ES
                  SLSQP(op0, space),  # Sequential Least Squares Programming
                  TNC(op0, space)]:  # Truncated Newton Method
    # For each algorithm, first configure and then execute one run.
    with Execution().set_objective(f)\
            .set_solution_space(space)\
            .set_max_fes(1000)\
            .set_rand_seed(1234)\
            .set_algorithm(algorithm)\
            .execute() as p:  # Execute the algorithm and get result.
        p.get_copy_of_best_x(b)  # Get a copy of the best solution.
        print(f"{algorithm} reaches {p.get_best_f()} with {space.to_str(b)}"
              f" at FE {p.get_last_improvement_fe()}")

# If you want to do more runs and get more detailed log information, check
# examples log_file_jssp.py, experiment_own_algorithm_and_problem.py,
# progress_plot.py, or experiment_2_algorithms_4_problems.py.
