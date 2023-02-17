"""Functions that can be used to test multi-objective algorithms."""
from math import inf, isfinite
from typing import Any, Final

import numpy as np
from numpy import array_equal
from numpy.random import Generator, default_rng

from moptipy.api.algorithm import (
    Algorithm0,
    Algorithm1,
    Algorithm2,
    check_algorithm,
)
from moptipy.api.encoding import Encoding
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_archive import MOArchivePruner
from moptipy.api.mo_execution import MOExecution
from moptipy.api.mo_problem import MOProblem
from moptipy.api.operators import check_op0, check_op1, check_op2
from moptipy.api.space import Space
from moptipy.mo.archive.keep_farthest import KeepFarthest
from moptipy.tests.component import validate_component
from moptipy.tests.encoding import validate_encoding
from moptipy.tests.mo_problem import validate_mo_problem
from moptipy.tests.space import validate_space
from moptipy.utils.nputils import rand_seed_generate
from moptipy.utils.types import check_int_range, type_error


def validate_mo_algorithm(
        algorithm: MOAlgorithm,
        solution_space: Space,
        problem: MOProblem,
        search_space: Space | None = None,
        encoding: Encoding | None = None,
        max_fes: int = 100,
        is_encoding_deterministic: bool = True) -> None:
    """
    Check whether a multi-objective algorithm follows the moptipy API.

    :param algorithm: the algorithm to test
    :param solution_space: the solution space
    :param problem: the problem to solve
    :param search_space: the optional search space
    :param encoding: the optional encoding
    :param max_fes: the maximum number of FEs
    :param is_encoding_deterministic: is the encoding deterministic?
    :raises TypeError: if `algorithm` is not a
        :class:`~moptipy.api.mo_algorithm.MOAlgorithm` instance
    :raises ValueError: if `algorithm` does not behave like it should
    """
    if not isinstance(algorithm, MOAlgorithm):
        raise type_error(algorithm, "algorithm", MOAlgorithm)

    check_algorithm(algorithm)
    if isinstance(algorithm, Algorithm0):
        check_op0(algorithm.op0)
        if isinstance(algorithm, Algorithm1):
            check_op1(algorithm.op1)
            if isinstance(algorithm, Algorithm2):
                check_op2(algorithm.op2)

    validate_component(algorithm)
    validate_mo_problem(problem, None, None)
    validate_space(solution_space, None)

    if encoding is not None:
        validate_encoding(encoding, None, None, None,
                          is_encoding_deterministic)
        validate_space(search_space, None)

    check_int_range(max_fes, "max_fes", 1, 1_000_000_000)
    lb: Final[int | float] = problem.lower_bound()
    if (not isfinite(lb)) and (lb != -inf):
        raise ValueError(f"objective lower bound cannot be {lb}.")
    ub = problem.upper_bound()
    if (not isfinite(ub)) and (ub != inf):
        raise ValueError(f"objective upper bound cannot be {ub}.")

    exp = MOExecution()
    exp.set_algorithm(algorithm)
    exp.set_max_fes(max_fes)
    exp.set_solution_space(solution_space)
    exp.set_objective(problem)
    if search_space is not None:
        exp.set_search_space(search_space)
        exp.set_encoding(encoding)

    random: Final[Generator] = default_rng()
    max_archive_size: Final[int] = int(random.integers(
        1, 1 << int(random.integers(1, 6))))
    exp.set_archive_max_size(max_archive_size)
    exp.set_archive_pruning_limit(
        max_archive_size + int(random.integers(0, 8)))
    if random.integers(2) <= 0:
        choice: int = int(random.integers(2))
        pruner: MOArchivePruner
        if choice <= 0:
            lst: list[int]
            while True:
                lst = [i for i in range(problem.f_dimension())
                       if random.integers(2) <= 0]
                if len(lst) > 0:
                    break
            pruner = KeepFarthest(problem, lst)
        else:
            pruner = MOArchivePruner()
        exp.set_archive_pruner(pruner)

    seed: Final[int] = rand_seed_generate(random)
    exp.set_rand_seed(seed)

    l_consumed_fes: int = -1
    l_last_improvement_fe: int = -1
    l_res_f: int | float = inf
    l_y: Any = None
    l_fs: np.ndarray | None = None
    l_x: Any = None
    for is_check in [False, True]:
        with exp.execute() as process:
            # re-raise any exception that was caught
            if hasattr(process, "_caught"):
                error = getattr(process, "_caught")
                if error is not None:
                    raise error
            # no exception? ok, let's check the data
            if not process.has_best():
                raise ValueError(
                    "The algorithm did not produce any solution.")

            if not process.should_terminate():
                raise ValueError(
                    "The algorithm stopped before hitting the "
                    "termination criterion.")
            consumed_fes: int = check_int_range(
                process.get_consumed_fes(), "consumed_fes", 1, max_fes)
            if is_check:
                if consumed_fes != l_consumed_fes:
                    raise ValueError(
                        f"consumed FEs changed from {l_consumed_fes} to "
                        f"{consumed_fes} in second run for seed {seed}")
            else:
                l_consumed_fes = consumed_fes
            last_improvement_fe = process.get_last_improvement_fe()
            check_int_range(last_improvement_fe, "last_improvement_fe",
                            1, consumed_fes)
            if is_check:
                if last_improvement_fe != l_last_improvement_fe:
                    raise ValueError(
                        "last improvement FEs changed from "
                        f"{l_last_improvement_fe} to {last_improvement_fe} in"
                        f" second run for seed {seed}")
            else:
                l_last_improvement_fe = last_improvement_fe

            consumed_time: int = check_int_range(
                process.get_consumed_time_millis(), "consumed_time",
                0, 100_0000_000)
            check_int_range(
                process.get_last_improvement_time_millis(),
                "last_improvement_time", 0, consumed_time)

            if lb != process.lower_bound():
                raise ValueError(
                    "Inconsistent lower bounds between process "
                    f"({process.lower_bound()}) and scalarized "
                    f"objective ({lb}).")
            if ub != process.upper_bound():
                raise ValueError(
                    "Inconsistent upper bounds between process "
                    f"({process.upper_bound()}) and scalarized "
                    f"objective ({ub}).")

            res_f: float | int = process.get_best_f()
            if not isfinite(res_f):
                raise ValueError(
                    "Infinite scalarized objective value of result.")
            if (res_f < lb) or (res_f > ub):
                raise ValueError(
                    f"Objective value {res_f} outside of bounds [{lb},{ub}].")
            if is_check:
                if res_f != l_res_f:
                    raise ValueError(
                        f"result f changed from {l_res_f} to {res_f} in"
                        f" second run for seed {seed}")
            else:
                l_res_f = res_f

            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)
            fs1 = problem.f_create()
            fs2 = problem.f_create()
            process.get_copy_of_best_y(y)
            check_f = problem.f_evaluate(y, fs1)
            if check_f != res_f:
                raise ValueError(
                    f"Inconsistent objective value {res_f} from process "
                    f"compared to {check_f} from objective function.")
            process.get_copy_of_best_fs(fs2)
            if not array_equal(fs1, fs2):
                raise ValueError(
                    f"Inconsistent objective vectors {fs1} and {fs2}.")

            if is_check:
                if not solution_space.is_equal(y, l_y):
                    raise ValueError(f"solution changed from {l_y} to {y} in "
                                     f"the second run of seed {seed}")
                if res_f != l_res_f:
                    raise ValueError(
                        f"result f changed from {l_res_f} to {res_f} in the "
                        f"second run of seed {seed}")
                if not np.array_equal(fs1, l_fs):
                    raise ValueError(
                        f"result fs changed from {l_fs} to {fs1} in the "
                        f"second run of seed {seed}")
            else:
                l_y = y
                l_fs = fs1
                l_res_f = res_f

            x: Any | None = None
            if search_space is not None:
                x = search_space.create()
                process.get_copy_of_best_x(x)
                search_space.validate(x)

                if is_check:
                    if not search_space.is_equal(x, l_x):
                        raise ValueError(
                            f"result x changed from {l_x} to {x} in the "
                            f"second run of seed {seed}")
                else:
                    l_x = x

            if encoding is not None:
                y2 = solution_space.create()
                encoding.decode(x, y2)
                solution_space.validate(y2)
                if is_encoding_deterministic:
                    solution_space.is_equal(y, y2)
