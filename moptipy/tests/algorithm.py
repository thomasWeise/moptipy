"""Functions that can be used to test algorithm implementations."""
from math import inf, isfinite
from typing import Any, Callable, Final

from moptipy.api.algorithm import (
    Algorithm,
    Algorithm0,
    Algorithm1,
    Algorithm2,
    check_algorithm,
)
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.operators import check_op0, check_op1, check_op2
from moptipy.api.space import Space
from moptipy.tests.component import validate_component
from moptipy.tests.encoding import validate_encoding
from moptipy.tests.objective import validate_objective
from moptipy.tests.space import validate_space
from moptipy.utils.nputils import rand_seed_generate
from moptipy.utils.types import check_int_range, type_error


def validate_algorithm(algorithm: Algorithm,
                       solution_space: Space,
                       objective: Objective,
                       search_space: Space | None = None,
                       encoding: Encoding | None = None,
                       max_fes: int = 100,
                       required_result: int | float | None = None,
                       uses_all_fes_if_goal_not_reached: bool = True,
                       is_encoding_deterministic: bool = True,
                       post: Callable[[Algorithm, int], Any] | None = None) \
        -> None:
    """
    Check whether an algorithm follows the moptipy API specification.

    :param algorithm: the algorithm to test
    :param solution_space: the solution space
    :param objective: the objective function
    :param search_space: the optional search space
    :param encoding: the optional encoding
    :param max_fes: the maximum number of FEs
    :param required_result: the optional required result quality
    :param uses_all_fes_if_goal_not_reached: will the algorithm use all FEs
        unless it reaches the goal?
    :param is_encoding_deterministic: is the encoding deterministic?
    :param post: a check to run after each execution of the algorithm,
        receiving the algorithm and the number of consumed FEs as parameter
    :raises TypeError: if `algorithm` is not a
        :class:`~moptipy.api.algorithm.Algorithm` instance
    :raises ValueError: if `algorithm` does not behave like it should
    """
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "algorithm", Algorithm)
    if (post is not None) and (not callable(post)):
        raise type_error(post, "post", None, call=True)

    check_algorithm(algorithm)
    if isinstance(algorithm, Algorithm0):
        check_op0(algorithm.op0)
        if isinstance(algorithm, Algorithm1):
            check_op1(algorithm.op1)
            if isinstance(algorithm, Algorithm2):
                check_op2(algorithm.op2)

    validate_component(algorithm)
    validate_space(solution_space, None)
    validate_objective(objective, None, None)
    if encoding is not None:
        validate_encoding(encoding, None, None, None,
                          is_encoding_deterministic)
        validate_space(search_space, None)

    check_int_range(max_fes, "max_fes", 1, 1_000_000_000)

    exp = Execution()
    exp.set_algorithm(algorithm)
    exp.set_max_fes(max_fes)
    exp.set_solution_space(solution_space)
    exp.set_objective(objective)
    seed: Final[int] = rand_seed_generate()
    if not isinstance(seed, int):
        raise type_error(seed, "seed", int)
    if not (0 <= seed <= 18446744073709551615):
        raise ValueError(f"invalid seed={seed}.")
    exp.set_rand_seed(seed)
    if search_space is not None:
        exp.set_search_space(search_space)
        exp.set_encoding(encoding)

    lb: Final[int | float] = objective.lower_bound()
    if (not isfinite(lb)) and (lb != -inf):
        raise ValueError(f"objective lower bound cannot be {lb}"
                         f" for {algorithm} on objective {objective}.")
    ub = objective.upper_bound()
    if (not isfinite(ub)) and (ub != inf):
        raise ValueError(f"objective upper bound cannot be {ub}"
                         f" for {algorithm} on objective {objective}.")

    if required_result is not None:
        if not (lb <= required_result <= ub):
            raise ValueError(f"required result must be in [{lb},{ub}], "
                             f"for {algorithm} on {objective} but "
                             f"is {required_result}")
        if (not isfinite(required_result)) and (required_result != -inf):
            raise ValueError(f"required_result must not be {required_result} "
                             f"for {algorithm} on {objective}.")

    progress: Final[tuple[list[int | float], list[int | float]]] = \
        [], []  # the progrss lists
    evaluate: Final[Callable[[Any], int | float]] = objective.evaluate

    for index in range(2 if is_encoding_deterministic else 1):

        if is_encoding_deterministic:

            def __k(xy, ii=index, ev=evaluate, pp=progress) -> int | float:
                rr = ev(xy)
                pp[ii].append(rr)
                return rr

            objective.evaluate = __k  # type: ignore

        with exp.execute() as process:
            # re-raise any exception that was caught
            if hasattr(process, "_caught"):
                error = getattr(process, "_caught")
                if error is not None:
                    raise error
            # no exception? ok, let's check the data
            if not process.has_best():
                raise ValueError(f"The algorithm {algorithm} did not produce "
                                 f"any solution on {objective} and "
                                 f"seed {seed}.")

            if (not process.should_terminate()) \
                    and uses_all_fes_if_goal_not_reached:
                raise ValueError(f"The algorithm {algorithm} stopped "
                                 f"before hitting the termination "
                                 f"criterion on {objective} and seed {seed}.")

            consumed_fes: int = check_int_range(
                process.get_consumed_fes(), "consumed_fes", 1, max_fes)
            last_imp_fe: int = check_int_range(
                process.get_last_improvement_fe(),
                "last_improvement_fe", 1, consumed_fes)
            consumed_time: int = check_int_range(
                process.get_consumed_time_millis(), "consumed_time",
                0, 100_0000_000)
            last_imp_time: int = check_int_range(
                process.get_last_improvement_time_millis(),
                "last_improvement_time", 0, consumed_time)
            if lb != process.lower_bound():
                raise ValueError(
                    "Inconsistent lower bounds between process "
                    f"({process.lower_bound()}) and objective ({lb})"
                    f" for {algorithm} on {objective} and seed {seed}.")
            if ub != process.upper_bound():
                raise ValueError(
                    "Inconsistent upper bounds between process "
                    f"({process.upper_bound()}) and objective ({ub}) "
                    f" for {algorithm} on {objective} and seed {seed}.")

            res_f: float | int = process.get_best_f()
            if not isfinite(res_f):
                raise ValueError(f"Infinite objective value of result "
                                 f"for {algorithm} on {objective}.")
            if (res_f < lb) or (res_f > ub):
                raise ValueError(f"Objective value {res_f} outside of bounds "
                                 f"[{lb},{ub}] for {algorithm} on "
                                 f"{objective} and seed {seed}.")

            if (required_result is not None) and (res_f > required_result):
                raise ValueError(
                    f"Algorithm {algorithm} should find solution of "
                    f"quality {required_result} on {objective}, but got "
                    f"one of {res_f} and seed {seed}.")

            if res_f <= lb:
                if last_imp_fe != consumed_fes:
                    raise ValueError(
                        f"if result={res_f} is as good as lb={lb}, then "
                        f"last_imp_fe={last_imp_fe} must equal"
                        f" consumed_fe={consumed_fes} for {algorithm} on "
                        f"{objective} and seed {seed}.")
                if (10_000 + (1.05 * last_imp_time)) < consumed_time:
                    raise ValueError(
                        f"if result={res_f} is as good as lb={lb}, then "
                        f"last_imp_time={last_imp_time} must not be much less"
                        f" than consumed_time={consumed_time} for "
                        f"{algorithm} on {objective} and seed {seed}.")
            elif uses_all_fes_if_goal_not_reached \
                    and (consumed_fes != max_fes):
                raise ValueError(
                    f"if result={res_f} is worse than lb={lb}, then "
                    f"consumed_fes={consumed_fes} must equal "
                    f"max_fes={max_fes} for {algorithm} on {objective}"
                    f" and seed {seed}.")

            y = solution_space.create()
            process.get_copy_of_best_y(y)
            solution_space.validate(y)
            check_f = objective.evaluate(y)
            if check_f != res_f:
                raise ValueError(
                    f"Inconsistent objective value {res_f} from process "
                    f"compared to {check_f} from objective function for "
                    f"{algorithm} on {objective} and seed {seed}.")

            x: Any | None = None
            if search_space is not None:
                x = search_space.create()
                process.get_copy_of_best_x(x)
                search_space.validate(x)

            if encoding is not None:
                y2 = solution_space.create()
                encoding.decode(x, y2)
                solution_space.validate(y2)
                if is_encoding_deterministic \
                        and not solution_space.is_equal(y, y2):
                    raise ValueError(
                        f"error when mapping point in search space {x} to "
                        f"solution {y2}, because it should be {y} for "
                        f"{algorithm} on {objective} under "
                        f"encoding {encoding} and seed {seed}")

            if post is not None:
                post(algorithm, consumed_fes)

    objective.evaluate = evaluate  # type: ignore

    if is_encoding_deterministic and (progress[0] != progress[1]):
        raise ValueError(f"when applying algorithm {algorithm} to "
                         f"{objective} under encoding {encoding} twice "
                         f"with the same seed {seed} did lead to different "
                         f"runs!")
