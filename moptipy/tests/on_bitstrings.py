"""Test stuff on bit strings."""

from typing import Callable, Union, Iterable, List, Optional, Dict, Any, \
    cast, Final, Set

import numpy as np
from numpy.random import default_rng, Generator

from moptipy.algorithms.so.fitness import Fitness
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.execution import Execution
from moptipy.api.mo_algorithm import MOAlgorithm
from moptipy.api.mo_problem import MOProblem
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.examples.bitstrings.ising1d import Ising1d
from moptipy.examples.bitstrings.leadingones import LeadingOnes
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.examples.bitstrings.zeromax import ZeroMax
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.fitness import validate_fitness
from moptipy.tests.mo_algorithm import validate_mo_algorithm
from moptipy.tests.op0 import validate_op0
from moptipy.tests.op1 import validate_op1
from moptipy.tests.op2 import validate_op2
from moptipy.utils.nputils import array_to_str
from moptipy.utils.types import type_error, type_name_of


def dimensions_for_tests() -> Iterable[int]:
    """
    Get a sequence of dimensions for tests.

    :returns: the sequence of integers
    """
    r = default_rng()
    bs: List[int] = [1, 2, 3, 4, 5, 10, 16, 100,
                     int(r.integers(20, 50)), int(r.integers(200, 300))]
    r.shuffle(cast(List, bs))
    return bs


def bitstrings_for_tests() -> Iterable[BitStrings]:
    """
    Get a sequence of bit strings for tests.

    :returns: the sequence of BitStrings
    """
    return [BitStrings(i) for i in dimensions_for_tests()]


def random_bit_string(random: Generator, x: np.ndarray) -> np.ndarray:
    """
    Randomize a bit string.

    :param random: the random number generator
    :param x: the bit string
    :returns: the array
    """
    ri = random.integers
    for i in range(len(x)):  # pylint: disable=C0200
        x[i] = ri(2) <= 0
    return x


def validate_op0_on_1_bitstrings(
        op0: Union[Op0, Callable[[BitStrings], Op0]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on one bit strings instance.

    :param op0: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op0": op0(search_space) if callable(op0) else op0,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op0(**args)


def validate_op0_on_bitstrings(
        op0: Union[Op0, Callable[[BitStrings], Op0]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on several BitStrings instances.

    :param op0: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op0_on_1_bitstrings(op0, bst,
                                     number_of_samples, min_unique_samples)


def validate_op1_on_1_bitstrings(
        op1: Union[Op1, Callable[[BitStrings], Op1]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on one BitStrings instance.

    :param op1: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op1": op1(search_space) if callable(op1) else op1,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op1(**args)


def validate_op1_on_bitstrings(
        op1: Union[Op1, Callable[[BitStrings], Op1]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the unary operator on several BitStrings instances.

    :param op1: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op1_on_1_bitstrings(op1, bst,
                                     number_of_samples, min_unique_samples)


def validate_op2_on_1_bitstrings(
        op2: Union[Op2, Callable[[BitStrings], Op2]],
        search_space: BitStrings,
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the binary operator on one BitStrings instance.

    :param op2: the operator or operator factory
    :param search_space: the search space
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    args: Dict[str, Any] = {
        "op2": op2(search_space) if callable(op2) else op2,
        "search_space": search_space,
        "make_search_space_element_valid": random_bit_string
    }
    if number_of_samples is not None:
        args["number_of_samples"] = number_of_samples
    if min_unique_samples is not None:
        args["min_unique_samples"] = min_unique_samples
    validate_op2(**args)


def validate_op2_on_bitstrings(
        op2: Union[Op2, Callable[[BitStrings], Op2]],
        number_of_samples: Optional[int] = None,
        min_unique_samples:
        Optional[Union[int, Callable[[int, BitStrings], int]]]
        = None) -> None:
    """
    Validate the binary operator on several BitStrings instances.

    :param op2: the operator or operator factory
    :param number_of_samples: the optional number of samples
    :param min_unique_samples: the optional unique samples
    """
    for bst in bitstrings_for_tests():
        validate_op2_on_1_bitstrings(op2, bst,
                                     number_of_samples, min_unique_samples)


def validate_algorithm_on_bitstrings(
        objective: Union[Objective, Callable[[int], Objective]],
        algorithm: Union[Algorithm,
                         Callable[[BitStrings, Objective], Algorithm]],
        dimension: int = 5,
        max_fes: int = 100,
        required_result: Optional[Union[int, Callable[
            [int, int], int]]] = None) -> None:
    """
    Check the validity of a black-box algorithm on a bit strings problem.

    :param algorithm: the algorithm or algorithm factory
    :param objective: the objective function or function factory
    :param dimension: the dimension of the problem
    :param max_fes: the maximum number of FEs
    :param required_result: the optional required result quality
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, 'algorithm', Algorithm, True)
    if not (isinstance(objective, Objective) or callable(objective)):
        raise type_error(objective, "objective", Objective, True)
    if not isinstance(dimension, int):
        raise type_error(dimension, 'dimension', int)
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, but got {dimension}.")

    if callable(objective):
        objective = objective(dimension)
    if not isinstance(objective, Objective):
        raise type_error(objective, "result of callable 'objective'",
                         Objective)
    bs: Final[BitStrings] = BitStrings(dimension)
    if callable(algorithm):
        algorithm = algorithm(bs, objective)
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "result of callable 'algorithm'",
                         Algorithm)

    goal: Optional[int]
    if callable(required_result):
        goal = required_result(max_fes, dimension)
    else:
        goal = required_result

    validate_algorithm(algorithm=algorithm,
                       solution_space=bs,
                       objective=objective,
                       max_fes=max_fes,
                       required_result=goal)


def validate_algorithm_on_onemax(
        algorithm: Union[Algorithm, Callable[
            [BitStrings, Objective], Algorithm]]) -> None:
    """
    Check the validity of a black-box algorithm on OneMax.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    for i in dimensions_for_tests():
        rr: int
        if i < 3:
            rr = 1
        else:
            rr = max(1, i // 2, i - int(max_fes ** 0.5))
        validate_algorithm_on_bitstrings(
            objective=OneMax,
            algorithm=algorithm,
            dimension=i,
            max_fes=max_fes,
            required_result=rr)


def validate_algorithm_on_leadingones(
        algorithm: Union[
            Algorithm, Callable[[BitStrings, Objective], Algorithm]]) -> None:
    """
    Check the validity of a black-box algorithm on LeadingOnes.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    for i in dimensions_for_tests():
        rr: int
        if i < 3:
            rr = 0
        elif max_fes > (10 * (i ** 1.5)):
            rr = i - 1
        else:
            rr = i
        validate_algorithm_on_bitstrings(
            objective=LeadingOnes,
            algorithm=algorithm,
            dimension=i,
            max_fes=int(1.25 * max_fes),
            required_result=rr)


def validate_mo_algorithm_on_bitstrings(
        problem: Union[MOProblem, Callable[[int], MOProblem]],
        algorithm: Union[MOAlgorithm, Callable[
            [BitStrings, MOProblem], MOAlgorithm]],
        dimension: int = 5,
        max_fes: int = 100) -> None:
    """
    Check a black-box multi-objective algorithm on a bit strings problem.

    :param algorithm: the algorithm or algorithm factory
    :param problem: the multi-objective optimization problem or factory
    :param dimension: the dimension of the problem
    :param max_fes: the maximum number of FEs
    """
    if not (isinstance(algorithm, MOAlgorithm) or callable(algorithm)):
        raise type_error(algorithm, 'algorithm', MOAlgorithm, True)
    if not (isinstance(problem, MOProblem) or callable(problem)):
        raise type_error(problem, "problem", MOProblem, True)
    if not isinstance(dimension, int):
        raise type_error(dimension, 'dimension', int)
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, but got {dimension}.")

    if callable(problem):
        problem = problem(dimension)
    if not isinstance(problem, MOProblem):
        raise type_error(problem, "result of callable 'problem'",
                         MOProblem)
    bs: Final[BitStrings] = BitStrings(dimension)
    if callable(algorithm):
        algorithm = algorithm(bs, problem)
    if not isinstance(algorithm, MOAlgorithm):
        raise type_error(algorithm, "result of callable 'algorithm'",
                         MOAlgorithm)

    validate_mo_algorithm(algorithm=algorithm,
                          solution_space=bs,
                          problem=problem,
                          max_fes=max_fes)


def validate_mo_algorithm_on_2_bitstring_problems(
        algorithm: Union[MOAlgorithm, Callable[
            [BitStrings, MOProblem], MOAlgorithm]]) -> None:
    """
    Check the validity of a black-box algorithm on OneMax and ZeroMax.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    random: Final[Generator] = default_rng()
    for i in dimensions_for_tests():
        weights: List[Union[int, float]]
        if random.integers(2) <= 0:
            weights = [float(random.uniform(0.01, 10)),
                       float(random.uniform(0.01, 10))]
        else:
            weights = [1 + int(random.integers(1 << random.integers(40))),
                       1 + int(random.integers(1 << random.integers(40)))]
        validate_mo_algorithm_on_bitstrings(
            problem=WeightedSum([OneMax(i), ZeroMax(i)], weights),
            algorithm=algorithm,
            dimension=i,
            max_fes=max_fes)


def validate_mo_algorithm_on_3_bitstring_problems(
        algorithm: Union[MOAlgorithm, Callable[
            [BitStrings, MOProblem], MOAlgorithm]]) -> None:
    """
    Check the validity of an algorithm on OneMax, ZeroMax, and Ising1d.

    :param algorithm: the algorithm or algorithm factory
    """
    max_fes: Final[int] = 100
    random: Final[Generator] = default_rng()
    for i in dimensions_for_tests():
        weights: List[Union[int, float]]
        if random.integers(2) <= 0:
            weights = [float(random.uniform(0.01, 10)),
                       float(random.uniform(0.01, 10)),
                       float(random.uniform(0.01, 10))]
        else:
            weights = [1 + int(random.integers(1 << random.integers(40))),
                       1 + int(random.integers(1 << random.integers(40))),
                       1 + int(random.integers(1 << random.integers(40)))]
        validate_mo_algorithm_on_bitstrings(
            problem=WeightedSum([OneMax(i), ZeroMax(i), Ising1d(i)],
                                weights),
            algorithm=algorithm,
            dimension=i,
            max_fes=max_fes)


def verify_algorithms_equivalent(
        algorithms: Iterable[Callable[[BitStrings, Objective], Algorithm]]) \
        -> None:
    """
    Verify that a set of algorithms performs identical steps.

    :param algorithms: the sequence of algorithms
    """
    if not isinstance(algorithms, Iterable):
        raise type_error(algorithms, "algorithms", Iterable)

    random: Final[Generator] = default_rng()
    dim: Final[int] = int(random.integers(4, 16))
    space: Final[BitStrings] = BitStrings(dim)
    steps: Final[int] = int(random.integers(100, 1000))
    choice: Final[int] = int(random.integers(3))
    f: Final[Objective] = \
        LeadingOnes(dim) if choice <= 0 else \
        OneMax(dim) if choice <= 1 else \
        Ising1d(dim)
    evaluate: Final[Callable] = f.evaluate
    seed: Final[int] = int(random.integers(1 << 62))

    result1: Final[List[bool]] = []
    result2: Final[List[bool]] = []
    first: bool = True
    first_name: str = ""
    do_fes: int = -1
    do_res: Union[int, float] = -1
    index: int = -1
    for algo in algorithms:
        index += 1
        if not callable(algo):
            raise type_error(algo, f"algorithms[{index}] for {f}", call=True)
        algorithm: Algorithm = check_algorithm(algo(space, f))
        current_name: str = str(algorithm)
        if first:
            first_name = current_name
            result = result1
        else:
            result = result2
            result.clear()

        def ff(x) -> int:
            nonlocal result
            nonlocal evaluate
            rres = evaluate(x)
            result.extend(x)  # pylint: disable=W0640
            return rres

        ex = Execution()
        ex.set_algorithm(algorithm)
        ex.set_solution_space(space)
        f.evaluate = ff  # type: ignore
        ex.set_objective(f)
        ex.set_rand_seed(seed)
        ex.set_max_fes(steps)
        with ex.execute() as p:
            cf = p.get_consumed_fes()
            if not (0 < cf <= steps):
                raise ValueError(f"{current_name} consumed {cf} FS for "
                                 f"{steps} max FEs on {f} for seed {seed}.")
            if first:
                do_fes = cf
            elif do_fes != cf:
                raise ValueError(f"{current_name} consumed {cf} FEs but "
                                 f"{first_name} consumed {do_fes} FEs on "
                                 f"{f} for seed {seed}.")
            res = p.get_best_f()
            if not (0 <= res <= dim):
                raise ValueError(f"{current_name} got {res} as objective "
                                 f"value on {f} for seed {seed}.")
            if first:
                do_res = res
            elif do_res != res:
                raise ValueError(
                    f"{current_name} got {res} as objective value on {f} but "
                    f"{first_name} got {do_res} for seed {seed}.")
            if len(result) != (cf * dim):
                raise ValueError(
                    f"len(result) == {len(result)}, but should be {cf * dim} "
                    f"for {current_name} for seed {seed} on {f}.")
        if (not first) and (result1 != result2):
            raise ValueError(
                f"{current_name} produced different steps than {first_name} "
                f"on {f} for seed {seed}: is "
                f"{array_to_str(np.array(result2))}"
                f" but should be {array_to_str(np.array(result1))}.")

        first = False


def validate_fitness_on_bitstrings(
        fitness: Union[Fitness, Callable[[Objective], Fitness]],
        class_needed: Union[str, type] = Fitness,
        prepare_objective: Callable[[Objective], Objective] = lambda x: x) \
        -> None:
    """
    Validate a fitness assignment process on bit strings.

    :param fitness: the fitness assignment process, or a callable creating it
    :param class_needed: the required class
    :param prepare_objective: prepare the objective function
    """
    if not isinstance(fitness, Fitness):
        if not callable(fitness):
            raise type_error(fitness, "fitness", Fitness, call=True)
    if not isinstance(class_needed, (str, type)):
        raise type_error(class_needed, "class_needed", (str, type))
    if not callable(prepare_objective):
        raise type_error(prepare_objective, "prepare_objective", call=True)

    random: Final[Generator] = default_rng()
    sizes: Set[int] = set()
    while len(sizes) < 4:
        sizes.add(int(random.integers(2, 10)))
    op0: Op0Random = Op0Random()
    for s in sizes:
        space: BitStrings = BitStrings(s)
        f: Objective = OneMax(s) if random.integers(2) <= 0 else LeadingOnes(s)
        f2 = prepare_objective(f)
        if not isinstance(f2, Objective):
            raise type_error(f2, f"prepare_objective({f})", Objective)
        del f
        if callable(fitness):
            ff = fitness(f2)
            if not isinstance(ff, Fitness):
                raise type_error(ff, f"fitness({f2})", Fitness)
            if isinstance(class_needed, str):
                name = type_name_of(ff)
                if name != class_needed:
                    raise TypeError(f"fitness assignment process should be "
                                    f"'{class_needed}', but is '{name}'.")
            elif not isinstance(ff, class_needed):
                raise type_error(ff, f"fitness({f2})", class_needed)
        else:
            ff = fitness
        validate_fitness(ff, f2, space, op0)
