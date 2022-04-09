"""Perform tests on the Job Shop Scheduling Problem."""

from typing import Callable, Optional, Union, Iterable, List, cast

from numpy.random import default_rng

from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.spaces.permutations import Permutations
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.objective import validate_objective
from moptipy.utils.types import type_error


def jssp_instances_for_tests() -> Iterable[str]:
    """
    Get a sequence of JSSP instances to test on.

    :returns: an iterable of JSSP instance names
    """
    r = default_rng()
    ri = r.integers
    insts: List[str] = [
        "demo", "ft06", "ft10", f"abz{ri(5, 10)}", f"dmu{ri(10, 81)}",
        f"orb0{ri(1, 10)}", f"swv{ri(10, 21)}", f"ta{ri(10, 65)}",
        f"ta{ri(65, 70)}", f"ta{ri(70, 75)}", f"yn{ri(1, 5)}"]
    r.shuffle(cast(List, insts))
    return insts


def make_gantt_valid(inst: Instance) -> Callable[[Gantt], Gantt]:
    """
    Make a function that creates valid Gantt charts.

    :param inst: the JSSP instance
    :returns: a function that can make gantt charts valid
    """
    pr = Permutations.with_repetitions(inst.jobs, inst.machines)
    op0 = Op0Shuffle(pr)
    oe = OperationBasedEncoding(inst)
    zrnd = default_rng()

    def __make_valid(x: Gantt, ppr=pr, pop0=op0, poe=oe, prnd=zrnd) -> Gantt:
        xx = ppr.create()
        pop0.op0(prnd, xx)
        poe.map(xx, x)
        return x

    return __make_valid


def validate_algorithm_on_1_jssp(
        algorithm: Union[Algorithm,
                         Callable[[Instance, Permutations], Algorithm]],
        instance: Optional[str] = None,
        max_fes: int = 100,
        required_result: Optional[int] = None) -> None:
    """
    Check the validity of a black-box algorithm on the JSSP.

    :param algorithm: the algorithm or algorithm factory
    :param instance: the instance name, or `None` to randomly pick one
    :param max_fes: the maximum number of FEs
    :param required_result: the optional required result quality
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, "algorithm", Algorithm, True)

    if instance is None:
        instance = str(default_rng().choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise ValueError("JSSP instance must either be a string or none, "
                         f"but is a {type(instance)}.")
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise ValueError(
            f"Error when loading JSSP instance '{instance}', "
            f"obtained {type(inst)} instead.")

    search_space = Permutations.with_repetitions(inst.jobs,
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

    goal: int
    if required_result is None:
        lb: int = objective.lower_bound()
        ub: int = objective.upper_bound()
        goal = max(lb + 1, min(ub - 1, int(0.5 + (lb + (0.96 * (ub - lb))))))
    else:
        goal = required_result

    validate_algorithm(algorithm=algorithm,
                       solution_space=solution_space,
                       objective=objective,
                       search_space=search_space,
                       encoding=encoding,
                       max_fes=max_fes,
                       required_result=goal)


def validate_algorithm_on_jssp(
        algorithm: Callable[[Instance, Permutations], Algorithm]) -> None:
    """
    Validate an algorithm on a set of JSSP instances.

    :param algorithm: the algorithm factory
    """
    for i in jssp_instances_for_tests():
        validate_algorithm_on_1_jssp(algorithm, i, 100)


def validate_objective_on_1_jssp(
        objective: Union[Objective, Callable[[Instance], Objective]],
        instance: Optional[str] = None,
        is_deterministic: bool = True) -> None:
    """
    Validate an objective function on 1 JSSP instance.

    :param objective: the objective function or a factory creating it
    :param instance: the instance name
    :param is_deterministic: is the objective function deterministic?
    """
    if instance is None:
        instance = str(default_rng().choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise ValueError("JSSP instance must either be a string or none, "
                         f"but is a {type(instance)}.")
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise ValueError(
            f"Error when loading JSSP instance '{instance}', "
            f"obtained {type(inst)} instead.")

    if callable(objective):
        objective = objective(inst)

    validate_objective(
        objective=objective,
        solution_space=GanttSpace(inst),
        make_solution_space_element_valid=make_gantt_valid(inst),
        is_deterministic=is_deterministic)


def validate_objective_on_jssp(
        objective: Union[Objective, Callable[[Instance], Objective]],
        is_deterministic: bool = True) -> None:
    """
    Validate an objective function on JSSP instances.

    :param objective: the objective function or a factory creating it
    :param is_deterministic: is the objective function deterministic?
    """
    for i in jssp_instances_for_tests():
        validate_objective_on_1_jssp(objective, i, is_deterministic)
