"""Run the moptipy example experiment."""
import os.path as pp
from math import factorial
from typing import Tuple, Dict, Set, List, Final, Iterable, Callable, \
    Optional, Union, cast

import moptipy.api.experiment as ex
from moptipy.algorithms.hill_climber import HillClimber
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.random_walk import RandomWalk
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.pwr.op0_shuffle import Op0Shuffle
from moptipy.operators.pwr.op1_swap2 import Op1Swap2
from moptipy.spaces.permutationswr import PermutationsWithRepetitions

#: The default instances to be used in our experiment.
EXPERIMENT_INSTANCES: \
    Final[Tuple[str, str, str, str, str, str, str, str, str]] = \
    ('dmu16', 'dmu26', 'ft06', 'la01', 'la06', 'orb01', 'swv11',
     'ta71', 'yn1')

#: The number of runs per instance in our experiment.
EXPERIMENT_RUNS: Final[int] = 7

#: We will perform two minutes per run.
EXPERIMENT_RUNTIME_MS: Final[int] = 2 * 60 * 1000


def propose_instances(n: int) -> Tuple[str, ...]:
    """
    Propose a set of instances to be used for our experiment.

    This function was used to obtain `EXPERIMENT_INSTANCES`. For
    `n=len(EXPERIMENT_INSTANCES)`, we now short-circuited it to
    `EXPERIMENT_INSTANCES`.

    :param int n: the number of instances to be proposed
    :return: a tuple with the instance names
    :rtype: str
    """
    if n == len(EXPERIMENT_INSTANCES):
        return EXPERIMENT_INSTANCES

    all_instance_names = Instance.list_resources()
    rm = str.maketrans("", "", "0123456789")
    prefixes: Set[str] = {s.translate(rm) for s in all_instance_names}
    purge = {"demo"}  # the forbidden instances
    chosen: Set[str] = set()  # the result instances
    prefixes.difference_update(purge)

    while len(chosen) < n:
        purge = purge.union(chosen)

        # collect all different problem scales that exist
        scales: Dict[Tuple[int, int, int, float], List[str]] = dict()
        groups: Dict[str, Set[Tuple[int, int, int, float]]] = dict()
        for inst_name in all_instance_names:
            if inst_name in purge:
                continue
            inst = Instance.from_resource(inst_name)

            # We consider the number of machines and jobs, but also the
            # the number of possible Gantt charts and the ratio of machines
            # to jobs as feature of an instance.
            size = inst.machines, inst.jobs, \
                factorial(inst.jobs) ** inst.machines, \
                inst.machines / inst.jobs

            # keep track of all sizes of a group
            group = inst_name.translate(rm)
            if group in groups:
                groups[group].add(size)
            else:
                ss: Set[Tuple[int, int, int, float]] = set()
                ss.add(size)
                groups[group] = ss

            # add instance to scale set
            if size in scales:
                lst = scales[size]
                if any(s.startswith(group) for s in lst):
                    continue
                lst.append(inst_name)
            else:
                scales[size] = [inst_name]

        # sort the scale values
        for lst in scales.values():
            lst.sort()

        # pick one item from all instance types that only occur in one size
        changed = True
        while changed and (len(chosen) < n):
            changed = False
            for group, sizes in groups.items():
                if len(sizes) <= 1:
                    sze = next(iter(sizes))
                    if sze in scales:
                        sel = scales[sze]
                        del scales[sze]
                        for name in sel:
                            if name.startswith(group):
                                chosen.add(name)
                                break
                    del groups[group]
                    changed = True
                    break

        # fill set of chosen instances based on extreme properties
        while len(chosen) < n:
            sizez = list(scales.keys())
            sizez.sort()
            if len(sizez) <= 0:
                break
            chosen_sizes: List[Tuple[int, int, int, float]] = list()

            # for each feature dimension, select maximum and minimum
            for dim in range(4):
                dim_min = min(sc[dim] for sc in sizez)
                for sc in sizez:
                    if sc[dim] <= dim_min:
                        if sc not in chosen_sizes:
                            chosen_sizes.append(sc)
                        break
                dim_max = max(sc[dim] for sc in sizez)
                for sc in sizez:
                    if sc[dim] >= dim_max:
                        if sc not in chosen_sizes:
                            chosen_sizes.append(sc)
                        break

            new_len = len(scales)
            min_size = 1
            while len(chosen) < n:
                old_len = new_len
                for scale in chosen_sizes:
                    choices = scales[scale]
                    if len(choices) <= min_size:
                        sel2 = choices[0]
                        group = sel2.translate(rm)
                        chosen_sizes.remove(scale)
                        del scales[scale]
                        chosen.add(sel2)
                        min_size = 1
                        for value in scales.values():
                            if len(value) > 1:
                                for i, v in enumerate(value):
                                    if v.startswith(group):
                                        del value[i]
                                        break
                        break
                new_len = len(chosen_sizes)
                if new_len >= old_len:
                    min_size += 1
                if new_len <= 0:
                    break

    # finalize
    result = list(chosen)
    result.sort()
    return tuple(result)


# noinspection PyUnusedLocal
def rs1(inst: Instance) -> SingleRandomSample:
    """
    Instantiate the single random sample algorithm.

    :param moptipy.examples.jssp.Instance inst: the jssp instance
    :return: the RandomSampling
    :rtype: moptipy.algorithms.RandomSampling
    """
    del inst
    return SingleRandomSample(Op0Shuffle())


# noinspection PyUnusedLocal
def rs(inst: Instance) -> RandomSampling:
    """
    Instantiate the random sampling.

    :param moptipy.examples.jssp.Instance inst: the jssp instance
    :return: the RandomSampling
    :rtype: moptipy.algorithms.RandomSampling
    """
    del inst
    return RandomSampling(Op0Shuffle())


# noinspection PyUnusedLocal
def hc_swap2(inst: Instance) -> HillClimber:
    """
    Instantiate the hill climber with 2-swap operator.

    :param moptipy.examples.jssp.Instance inst: the jssp instance
    :return: the HillClimber
    :rtype: moptipy.algorithms.HillClimber
    """
    del inst
    return HillClimber(Op0Shuffle(), Op1Swap2())


# noinspection PyUnusedLocal
def rw_swap2(inst: Instance) -> RandomWalk:
    """
    Instantiate the random walk with 2-swap operator.

    :param moptipy.examples.jssp.Instance inst: the jssp instance
    :return: the RandomWalk
    :rtype: moptipy.algorithms.RandomWalk
    """
    del inst
    return RandomWalk(Op0Shuffle(), Op1Swap2())


#: The default set of algorithms for our experiments
DEFAULT_ALGORITHMS: Final[Tuple[Callable, ...]] = \
    (rs1, rs, rw_swap2, hc_swap2)


def run_experiment(base_dir: str = pp.join(".", "results"),
                   algorithms: Iterable[Callable] = DEFAULT_ALGORITHMS,
                   instances: Iterable[str] = EXPERIMENT_INSTANCES,
                   n_runs: int = EXPERIMENT_RUNS,
                   max_time: Optional[int] = EXPERIMENT_RUNTIME_MS,
                   max_fes: Optional[int] = None) -> None:
    """
    Run the experiment.

    :param Iterable[Callable] algorithms: the algorithm factories
    :param Iterable[Callable] instances: the JSSP instance names
    :param str base_dir: the base directory
    :param int n_runs: the number of runs
    :param Optional[int] max_time: the maximum runtime in milliseconds
    :param Optional[int] max_fes: the maximum runtime in FEs
    """
    if not isinstance(base_dir, str):
        raise TypeError(
            f"base_dir must be str, but is {type(base_dir)}.")
    if not isinstance(algorithms, Iterable):
        raise TypeError(
            f"algorithms must be iterable, but is {type(algorithms)}.")
    if not isinstance(instances, Iterable):
        raise TypeError(
            f"Instances must be iterable, but is {type(instances)}.")
    if not isinstance(n_runs, int):
        raise TypeError(f"n_runs must be int but is {type(n_runs)}.")
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, but is {n_runs}.")
    if max_time is not None:
        if not isinstance(max_time, int):
            raise TypeError(f"max_time must be int but is {type(max_time)}.")
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, but is {max_time}.")
    if max_fes is not None:
        if not isinstance(max_fes, int):
            raise TypeError(f"max_fes must be int but is {type(max_fes)}.")
        if max_time <= 0:
            raise ValueError(f"max_fes must be positive, but is {max_fes}.")
    if (max_fes is None) and (max_time is None):
        raise ValueError("Either max_fes or max_time must be provided.")

    inst_gens = [(lambda s=s: Instance.from_resource(s)) for s in instances]
    if len(inst_gens) <= 0:
        raise ValueError("There must be at least one instance.")
    algo_gens = list()
    for algo in algorithms:
        def creator(inst: Instance, algor=algo):
            val: Union[Execution, Algorithm] = algor(inst)
            experiment: Execution
            if isinstance(val, Execution):
                experiment = cast(Execution, val)
            else:
                experiment = Execution()
                experiment.set_algorithm(cast(Algorithm, val))
            if max_time is not None:
                experiment.set_max_time_millis(max_time)
            if max_fes is not None:
                experiment.set_max_fes(max_fes)
            experiment.set_objective(Makespan(inst))
            experiment.set_solution_space(GanttSpace(inst))
            experiment.set_search_space(
                PermutationsWithRepetitions(inst.jobs, inst.machines))
            experiment.set_encoding(OperationBasedEncoding(inst))
            experiment.set_log_improvements()
            return experiment

        algo_gens.append(creator)
        del creator
    if len(algo_gens) <= 0:
        raise ValueError("There must be at least one algorithm.")

    ex.run_experiment(base_dir, inst_gens, algo_gens, n_runs)


# Execute experiment if run as script
if __name__ == '__main__':
    run_experiment()
