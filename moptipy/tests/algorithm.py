"""Functions that can be used to test algorithm implementations."""
from math import isfinite
from typing import Callable, Optional, Union

import numpy.random as rnd

from moptipy.api.algorithm import Algorithm, check_algorithm, Algorithm0, \
    Algorithm1, Algorithm2
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.api.operators import check_op0, check_op1, check_op2
from moptipy.api.space import Space
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.component import validate_component
from moptipy.tests.encoding import validate_encoding
from moptipy.tests.objective import validate_objective
from moptipy.tests.space import validate_space


def validate_algorithm(algorithm: Algorithm,
                       solution_space: Optional[Space] = None,
                       objective: Optional[Objective] = None,
                       search_space: Optional[Space] = None,
                       encoding: Optional[Encoding] = None,
                       max_fes: int = 100,
                       required_result: Optional[Union[int, float]] = None) \
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
    :raises TypeError: if `algorithm` is not an Algorithm instance
    :raises ValueError: if `algorithm` does not behave like it should
    """
    if not isinstance(algorithm, Algorithm):
        raise TypeError("Expected to receive an instance of Algorithm, but "
                        f"got a {type(algorithm)}.")

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
        validate_encoding(encoding, None, None, None)
        validate_space(search_space, None)

    if not isinstance(max_fes, int):
        raise ValueError(
            f"max_fes must be int but is '{type(max_fes)}'.")
    if max_fes <= 0:
        raise ValueError(f"max_fes must be > 0, but is {max_fes}.")

    exp = Execution()
    exp.set_algorithm(algorithm)
    exp.set_max_fes(max_fes)
    exp.set_solution_space(solution_space)
    exp.set_objective(objective)
    if not (search_space is None):
        exp.set_search_space(search_space)
        exp.set_encoding(encoding)

    with exp.execute() as process:

        if not process.has_current_best():
            raise ValueError("The algorithm did not produce any solution.")

        if not process.should_terminate():
            raise ValueError("The algorithm stopped before hitting the "
                             "termination criterion.")

        consumed_fes = process.get_consumed_fes()
        if not isinstance(consumed_fes, int):
            raise ValueError(
                f"Consumed FEs must be int, but are {type(consumed_fes)}.")
        if (consumed_fes <= 0) or (consumed_fes > max_fes):
            raise ValueError(
                f"Consumed FEs must be positive and <= {max_fes}, "
                f"but is {consumed_fes}.")

        last_imp_fe = process.get_last_improvement_fe()
        if not isinstance(last_imp_fe, int):
            raise ValueError("Last improvement FEs must be int, "
                             f"but are {type(last_imp_fe)}'.")
        if (last_imp_fe <= 0) or (last_imp_fe > consumed_fes):
            raise ValueError("Last improvement FEs must be positive and "
                             f"<= {consumed_fes}, but is {last_imp_fe}.")

        consumed_time = process.get_consumed_time_millis()
        if not isinstance(consumed_time, int):
            raise ValueError(
                f"Consumed time must be int, but is {type(consumed_time)}.")
        if consumed_time < 0:
            raise ValueError(
                f"Consumed time must be >= 0, but is {consumed_time}.")

        last_imp_time = process.get_last_improvement_time_millis()
        if not isinstance(last_imp_time, int):
            raise ValueError("Last improvement time must be int, "
                             f"but is {type(last_imp_time)}.")
        if (last_imp_time < 0) or (last_imp_time > consumed_time):
            raise ValueError(
                f"Consumed time must be >= 0 and <= {consumed_time}, but "
                f"is {last_imp_time}.")

        lb = objective.lower_bound()
        if lb != process.lower_bound():
            raise ValueError(
                "Inconsistent lower bounds between process "
                f"({process.lower_bound()}) and objective ({lb}).")

        ub = objective.upper_bound()
        if ub != process.upper_bound():
            raise ValueError(
                "Inconsistent upper bounds between process "
                f"({process.upper_bound()}) and objective ({ub}).")

        res_f = process.get_current_best_f()
        if not isfinite(res_f):
            raise ValueError("Infinite objective value of result.")
        if (res_f < lb) or (res_f > ub):
            raise ValueError(
                f"Objective value {res_f} outside of bounds [{lb},{ub}].")

        if required_result is not None:
            if res_f > required_result:
                raise ValueError(
                    "Algorithm should find solution of quality "
                    f"{required_result}, but got one of {res_f}.")

        y = solution_space.create()
        process.get_copy_of_current_best_y(y)
        solution_space.validate(y)
        check_f = objective.evaluate(y)
        if check_f != res_f:
            raise ValueError(
                f"Inconsistent objective value {res_f} from process compared "
                f"to {check_f} from objective function.")

        if search_space is not None:
            x = search_space.create()
            process.get_copy_of_current_best_x(x)
            search_space.validate(x)


def validate_algorithm_on_1_jssp(
        algorithm: Union[Algorithm,
                         Callable[[Instance, PermutationsWithRepetitions],
                                  Algorithm]],
        instance: Optional[str] = None,
        max_fes: int = 100,
        required_result: Optional[Union[int, float]] = None) -> None:
    """
    Check the validity of a black-box algorithm on the JSSP.

    :param Callable algorithm: the algorithm factory
    :param Optional[str] instance: the instance name, or `None` to randomly
        pick one
    :param int max_fes: the maximum number of FEs
    :param required_result: the optional required result quality
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise TypeError(
            "'algorithm' parameter must be an Algorithm or a callable that"
            "instantiates an algorithm for a given JSSP instance, but got a "
            f"{type(algorithm)} instead.")

    if instance is None:
        instance = str(rnd.default_rng().choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise ValueError("JSSP instance must either be a string or none, "
                         f"but is a {type(instance)}.")
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise ValueError(
            f"Error when loading JSSP instance '{instance}', "
            f"obtained {type(inst)} instead.")

    search_space = PermutationsWithRepetitions(inst.jobs,
                                               inst.machines)
    if callable(algorithm):
        algorithm = algorithm(inst, search_space)
    if not isinstance(algorithm, Algorithm):
        raise ValueError(
            "'algorithm' parameter must be a callable that instantiates "
            f"an algorithm for JSSP instance '{instance}', but it created a "
            f"'{type(algorithm)}' instead.")

    solution_space = GanttSpace(inst)
    encoding = OperationBasedEncoding(inst)
    objective = Makespan(inst)

    validate_algorithm(algorithm=algorithm,
                       solution_space=solution_space,
                       objective=objective,
                       search_space=search_space,
                       encoding=encoding,
                       max_fes=max_fes,
                       required_result=(objective.upper_bound() - 1)
                       if required_result is None else required_result)


def validate_algorithm_on_jssp(
        algorithm: Callable[[Instance, PermutationsWithRepetitions],
                            Algorithm]) -> None:
    """
    Validate an algorithm on a set of JSSP instances.

    :param Callable algorithm: the algorithm factory
    """
    validate_algorithm_on_1_jssp(algorithm, "demo", 200, 190)
    validate_algorithm_on_1_jssp(algorithm, "abz7", 100)
    validate_algorithm_on_1_jssp(algorithm, "la37", 100)
    validate_algorithm_on_1_jssp(algorithm, "orb07", 100)
    validate_algorithm_on_1_jssp(algorithm, "yn3", 100)
    validate_algorithm_on_1_jssp(algorithm, "swv13", 100)
    validate_algorithm_on_1_jssp(algorithm, "dmu66", 100)
    validate_algorithm_on_1_jssp(algorithm, "dmu73", 100)
    validate_algorithm_on_1_jssp(algorithm, "ta71", 100)
    validate_algorithm_on_1_jssp(algorithm, "ta69", 100)
