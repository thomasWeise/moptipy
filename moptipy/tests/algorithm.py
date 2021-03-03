from math import isfinite
from typing import Callable, Optional
import numpy.random as rnd
from moptipy.api.experiment import Experiment
# noinspection PyProtectedMember
from moptipy.api.algorithm import Algorithm, _check_algorithm
from moptipy.api.encoding import Encoding
from moptipy.api.objective import Objective
from moptipy.api.space import Space
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import JSSPInstance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.spaces.permutationswr import PermutationsWithRepetitions
from moptipy.tests.component import check_component
from moptipy.tests.encoding import check_encoding
from moptipy.tests.objective import check_objective
from moptipy.tests.space import check_space


def check_algorithm(algorithm: Algorithm = None,
                    solution_space: Space = None,
                    objective: Objective = None,
                    search_space: Optional[Space] = None,
                    encoding: Optional[Encoding] = None,
                    max_fes: int = 100):
    """
    Check whether an algorithm follows the moptipy API specification.
    :param algorithm: the algorithm to test
    :param solution_space: the solution space
    :param objective: the objective function
    :param search_space: the optional search space
    :param encoding: the optional encoding
    :param max_fes: the maximum number of FEs
    """

    if not isinstance(algorithm, Algorithm):
        raise ValueError("Expected to receive an instance of Space, but "
                         "got a '" + str(type(algorithm)) + "'.")

    _check_algorithm(algorithm)

    check_component(component=algorithm)
    check_space(solution_space, make_valid=None)
    check_objective(objective, create_valid=None)
    if not (encoding is None):
        check_encoding(encoding, make_search_space_valid=None)
        check_space(search_space, make_valid=None)

    if not isinstance(max_fes, int):
        raise ValueError(
            "max_fes must be int but is '" + str(type(max_fes)) + "'.")
    if max_fes <= 0:
        raise ValueError("max_fes must be > 0, but is" + str(max_fes) + ".")

    exp = Experiment()
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
            raise ValueError("Consumed FEs must be int, but are '"
                             + str(type(consumed_fes)) + "'.")
        if (consumed_fes <= 0) or (consumed_fes > max_fes):
            raise ValueError("Consumed FEs must be positive and <= "
                             + str(max_fes) + ", but is "
                             + str(consumed_fes) + ".")

        last_imp_fe = process.get_last_improvement_fe()
        if not isinstance(last_imp_fe, int):
            raise ValueError("Last improvement FEs must be int, but are '"
                             + str(type(last_imp_fe)) + "'.")
        if (last_imp_fe <= 0) or (last_imp_fe > consumed_fes):
            raise ValueError("Last improvement FEs must be positive and <= "
                             + str(consumed_fes) + ", but is "
                             + str(last_imp_fe) + ".")

        consumed_time = process.get_consumed_time_millis()
        if not isinstance(consumed_time, int):
            raise ValueError("Consumed time must be int, but is '"
                             + str(type(consumed_time)) + "'.")
        if consumed_time < 0:
            raise ValueError("Consumed time must be >= 0, but is "
                             + str(consumed_time) + ".")

        last_imp_time = process.get_last_improvement_time_millis()
        if not isinstance(last_imp_time, int):
            raise ValueError("Last improvement time must be int, but is '"
                             + str(type(last_imp_time)) + "'.")
        if (last_imp_time < 0) or (last_imp_time > consumed_time):
            raise ValueError("Consumed time must be >= 0 and <= "
                             + str(consumed_time) + ", but is "
                             + str(last_imp_time) + ".")

        lb = objective.lower_bound()
        if lb != process.lower_bound():
            raise ValueError(
                "Inconsistent lower bounds between process ("
                + str(process.lower_bound()) + ") and objective ("
                + str(lb) + ").")

        ub = objective.upper_bound()
        if ub != process.upper_bound():
            raise ValueError(
                "Inconsistent upper bounds between process ("
                + str(process.upper_bound()) + ") and objective ("
                + str(ub) + ").")

        res_f = process.get_current_best_f()
        if not isfinite(res_f):
            raise ValueError("Infinite objective value of result.")
        if (res_f < lb) or (res_f > ub):
            raise ValueError("Objective value " + str(res_f)
                             + " outside of bounds [" + str(lb) + ","
                             + str(ub) + "].")

        y = solution_space.create()
        process.get_copy_of_current_best_y(y)
        check_f = objective.evaluate(y)
        if check_f != res_f:
            raise ValueError("Inconsistent objective value " + str(res_f)
                             + " from process compared to " + str(check_f)
                             + " from objective function.")


def check_algorithm_on_jssp(algorithm: Callable = None,
                            instance: Optional[str] = None,
                            max_fes: int = 100):
    if not isinstance(algorithm, Callable):
        raise ValueError(
            "Must 'algorithm' parameter must be a callable that instantiates"
            "an algorithm for a given JSSP instance, but got a '"
            + str(type(algorithm)) + "' instead.")

    if instance is None:
        instance = str(rnd.default_rng().choice(JSSPInstance.list_resources()))
    if not isinstance(instance, str):
        raise ValueError("JSSP instance must either be a string or none, "
                         "but is a '" + str(type(instance)) + "'.")
    instance = JSSPInstance.from_resource(instance)
    if not isinstance(instance, JSSPInstance):
        raise ValueError("Error when loading JSSP instance, obtained '"
                         + str(type(instance)) + "' instead.")

    algorithm = algorithm(instance)
    if not isinstance(algorithm, Algorithm):
        raise ValueError(
            "Must 'algorithm' parameter must be a callable that instantiates"
            "an algorithm for a given JSSP instance, but it created a '"
            + str(type(algorithm)) + "' instead.")

    search_space = PermutationsWithRepetitions(instance.jobs,
                                               instance.machines)
    solution_space = GanttSpace(instance)
    encoding = OperationBasedEncoding(instance)
    objective = Makespan(instance)

    check_algorithm(algorithm=algorithm,
                    solution_space=solution_space,
                    objective=objective,
                    search_space=search_space,
                    encoding=encoding,
                    max_fes=max_fes)
