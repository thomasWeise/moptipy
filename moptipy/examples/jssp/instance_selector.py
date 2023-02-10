"""Code for selecting interesting instances for smaller-scale experiments."""
import multiprocessing as mp
from math import ceil, factorial, inf, log2, sqrt
from os import sched_getaffinity
from typing import Callable, Final

import numpy as np  # type: ignore
from numpy.random import Generator, RandomState  # type: ignore
from sklearn.cluster import SpectralClustering  # type: ignore

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.console import logger
from moptipy.utils.nputils import (
    DEFAULT_FLOAT,
    DEFAULT_INT,
    rand_generator,
    rand_seeds_from_str,
)
from moptipy.utils.types import check_int_range, type_error


def __can_solve_instance(inst: Instance, seed: int,
                         queue: mp.Queue) -> None:
    """
    Execute one run of a RLS for 2.1min on the instance, return makespan.

    :param inst: the jssp instance to check
    :param seed: the seed of this run
    :param queue: the queue for the return values
    """
    x_space: Final[Permutations] = Permutations.with_repetitions(
        inst.jobs, inst.machines)
    y_space: Final[GanttSpace] = GanttSpace(inst)
    g: Final[OperationBasedEncoding] = OperationBasedEncoding(inst)
    o0: Final[Op0Shuffle] = Op0Shuffle(x_space)
    o1: Final[Op1Swap2] = Op1Swap2()
    f: Final[Makespan] = Makespan(inst)
    a: Final[RLS] = RLS(o0, o1)
    goal: Final[int] = inst.makespan_lower_bound

    ex: Final[Execution] = Execution()
    ex.set_search_space(x_space)
    ex.set_solution_space(y_space)
    ex.set_encoding(g)
    ex.set_algorithm(a)
    ex.set_objective(f)
    ex.set_max_time_millis(126_000)
    ex.set_goal_f(goal)
    ex.set_rand_seed(seed)
    with ex.execute() as process:
        queue.put(process.get_best_f())


def __is_instance_too_easy(inst: Instance) -> bool:
    """
    Check if an instance is too easy to be of any interest.

    We perform 16 runs of a (1+1)-EA on an instance, granting 2.1 minutes
    of runtime to each of them. An instance is considered as easy if one of
    the following conditions is fulfilled:

    - it is solved often enough (lower bound reached often)
    - there are not enough different results
    - the mean result is too close to the lower bound
    - the worst result is not far enough from the lower bound
    - the worst result is too close to the best result

    If any of these conditions hold, we consider the instance as too easy to
    be of any interest for our educational experiment.

    :param inst: the jssp instance to check
    :returns: `True` if the instance is too easy to be interesting
    """
    n_runs: Final[int] = 16
    seeds: Final[list[int]] = rand_seeds_from_str(string=inst.name,
                                                  n_seeds=n_runs)
    queue: Final[mp.Queue] = mp.Queue()

    while len(seeds) > 0:
        processes = [mp.Process(target=__can_solve_instance,
                                args=(inst, seeds.pop(), queue))
                     for _ in range(min(len(seeds), __DEFAULT_N_THREADS))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    goal: Final[int] = inst.makespan_lower_bound
    num_solved: int = 0
    diff: set[int] = set()
    min_makespan: int = 1 << 62
    sum_makespans: int = 0
    max_makespan: int = 0

    for _ in range(n_runs):
        makespan: int = queue.get()
        diff.add(makespan)
        if makespan <= min_makespan:
            min_makespan = makespan
            if makespan <= goal:
                num_solved += 1
        sum_makespans += makespan
        if makespan > max_makespan:
            max_makespan = makespan

    mean_makespan: Final[float] = sum_makespans / n_runs
    mean_threshold: Final[float] = min(goal + 32.0,
                                       max(1.01 * goal, goal + 3))
    max_threshold: Final[float] = min(goal + 64.0,
                                      max(1.02 * goal, goal + 5))
    min_threshold: Final[float] = min(min_makespan + 48.0,
                                      max(1.015 * min_makespan,
                                          min_makespan + 4))
    n_diff: Final[int] = len(diff)
    diff_threshold: Final[int] = max(3, int(0.5 + ceil(log2(n_runs))))
    solved_threshold: Final[int] = int(ceil(n_runs / 3))

    result: Final[bool] = (num_solved >= solved_threshold) \
        or (mean_makespan < mean_threshold) \
        or (max_makespan < max_threshold) \
        or (min_threshold > max_makespan) \
        or (n_diff <= diff_threshold)
    logger(f"Instance {inst.name} is {'easy' if result else 'hard'}, "
           f"since {n_runs} runs of the (1+1)-EA had {n_diff} different "
           f"results (threshold {diff_threshold}), a best result of "
           f"{min_makespan}, a mean result of {mean_makespan} (threshold: "
           f"{mean_threshold}), a worst result of {max_makespan} "
           f"(thresholds: {max_threshold}, {min_threshold}), and "
           f"reached the makespan lower bound ({goal}) {num_solved} "
           f"times (threshold: {solved_threshold}).")
    return result


#: The default number of threads to be used
__DEFAULT_N_THREADS: Final[int] = max(1, min(len(sched_getaffinity(0)), 128))


def compute_instances_that_are_too_easy() -> tuple[str, ...]:
    """
    Compute the set of instances that are too easy to be of any interest.

    For our educational experiments, we need instances that are not always
    solved to optimality by simple algorithms. Otherwise, for example, the
    box plots of the end results will collapse to single lines. Then, it
    will also be impossible to really say which algorithm performs best on
    the instance.
    See the documentation of `__is_instance_too_easy` for the conditions that
    lead to the rejection of an instance

    :returns: the tuple of instance names that should not be included in
        the experiment
    """
    logger("We now test the instances whether they are too easy.")
    easy: Final[list[str]] = []
    for inst_name in Instance.list_resources():
        if __is_instance_too_easy(Instance.from_resource(inst_name)):
            easy.append(inst_name)
    easy.sort()
    logger(f"Finished the test, found {len(easy)} easy instances: {easy}.")
    return tuple(easy)


#: This is the set of instances that are too easy to be interesting and shall
#: therefore be skipped from our experiments.
#: For a documentation about what constitutes 'too easy', see
#: :func:`compute_instances_that_are_too_easy`
__TOO_EASY: Final[set[str]] = \
    {"demo", "dmu21", "dmu22", "dmu24", "dmu25", "dmu31", "dmu32", "dmu33",
     "dmu34", "dmu35", "ft06", "ft20", "la01", "la02", "la03", "la04",
     "la05", "la06", "la07", "la08", "la09", "la10", "la11", "la12", "la13",
     "la14", "la15", "la17", "la18", "la23", "la26", "la28", "la30", "la31",
     "la32", "la33", "la34", "la35", "swv16", "swv17", "swv18", "swv19",
     "swv20", "ta51", "ta52", "ta53", "ta54", "ta55", "ta56", "ta57", "ta58",
     "ta59", "ta60", "ta61", "ta63", "ta68", "ta69", "ta71", "ta72", "ta73",
     "ta74", "ta75", "ta76", "ta77", "ta78", "ta79", "ta80"}


def __get_instances() -> list[Instance]:
    """
    Get the instances.

    :return: a tuple of instances
    """
    return [Instance.from_resource(name) for name in
            Instance.list_resources() if name not in __TOO_EASY]


def __optimize_clusters(cluster_groups: tuple[tuple[int, ...], ...],
                        n_groups: int,
                        extreme_groups: tuple[tuple[tuple[int,
                                                          int], ...], ...],
                        random: Generator) -> tuple[int, ...]:
    """
    Try to find optimal cluster-to-group assignments.

    We perform hill climbing with restarts.

    :param cluster_groups: the groups per cluster
    :param n_groups: the number of groups
    :param extreme_groups: the extreme groups
    :param random: the random number generator
    :return: the selected groups
    """
    n: Final[int] = len(cluster_groups)
    current: Final[np.ndarray] = np.zeros(n, DEFAULT_INT)
    current_f: tuple[int, int, int, int, float]
    best: Final[np.ndarray] = np.zeros(n, DEFAULT_INT)
    best_f: tuple[int, int, int, int, float]
    total_best: Final[np.ndarray] = np.zeros(n, DEFAULT_INT)
    total_best_f: tuple[int, int, int, int, float] = (-1, -1, -1, -1, -1)
    run_last_improved: int = 1
    run_current: int = 0
    run_max_none_improved: Final[int] = 4
    step_max_none_improved: Final[int] = int((2 + (n_groups * n)) ** 2)

    done: Final[np.ndarray] = np.zeros(n_groups, DEFAULT_INT)  # noqa
    extremes: Final[set[int]] = set()  # noqa

    logger(f"Beginning to optimize the assignment of {len(cluster_groups)} "
           f"clusters to {n_groups} groups. The minimum groups are "
           f"{extreme_groups[0]}, the maximum groups are {extreme_groups[1]}."
           f" We permit {run_max_none_improved} runs without improvement "
           f"before termination and {step_max_none_improved} FEs without "
           "improvement before stopping a run.")

    def __f(sol: np.ndarray) -> tuple[int, int, int, int, float]:
        """
        Compute the quality: The internal objective function.

        We maximize five things:
        First, we maximize the number of group assignments that allow us to
        include the extreme instances in the result. These are the smallest
        and largest-scale instances.
        Second, we maximize the number of instance sets covered.
        Third, we maximize the minimum number of usages of the instance sets.
        Fourth, we minimize the maximum number of usages of the instance sets
        with the goal to balance the usages evenly over different sets.
        Fifth, we minimize the standard deviation of the instance set usages,
        again with the goal to balance the usages evenly over different sets.

        :param sol: the solution
        :return: the objective values
        """
        nonlocal done  # the number of uses per set
        nonlocal extreme_groups  # the tuple-tuple with the extreme groups
        nonlocal extremes  # the extreme groups already picked

        # first, we count how often each group has been used
        done.fill(0)
        for group in sol:
            done[group] += 1

        # second, we check which cluster-to-group assignment permits
        # using an extreme instance
        extremes.clear()
        for groups in extreme_groups:
            for group in groups:
                if (sol[group[0]] == group[1]) and (group[0] not in extremes):
                    extremes.add(group[0])
                    break
        return len(extremes), int(np.sum(done > 0)), done.min(), \
            -done.max(), -np.std(done)

    # The outer loop: the restarts of the hill climber.
    # We continue to execute runs until a given number of successive runs did
    # not improve the overall best solutions. Then we stop.
    while (run_current - run_last_improved) <= run_max_none_improved:
        run_current += 1

        # generate an initial best solution
        for i in range(n):
            cg = cluster_groups[i]
            best[i] = cg[random.integers(len(cg))]
        best_f = __f(best)  # compute quality
        use_f = best_f

        if best_f > total_best_f:  # accept solution if it is better
            total_best_f = best_f
            run_last_improved = run_current  # new global best
            np.copyto(total_best, best)
            logger(f"New global best {tuple(best)} with quality "
                   f"{best_f} found in run {run_current}.")
        else:
            logger(f"Run {run_current} starts with quality {best_f}.")

        # local search step
        # We continue the local search until a given number of successive
        # steps did not yield any improvement. Then we terminate the run.
        step_current: int = 0
        step_last_improved: int = 1
        while (step_current - step_last_improved) <= step_max_none_improved:
            step_current += 1
            np.copyto(current, best)
            while True:  # perform at least one move
                i = random.integers(n)
                cg = cluster_groups[i]
                lcg = len(cg)
                if lcg <= 1:
                    continue
                while True:  # perform the search move
                    new_v = cg[random.integers(lcg)]
                    if new_v != best[i]:
                        current[i] = new_v
                        break
                # 50/50 chance to perform more moves
                if random.integers(2) <= 0:
                    break

            current_f = __f(current)  # evaluate quality
            # accept if better or with small probability
            if (current_f >= use_f) or (random.integers(50) < 1):
                use_f = current_f
                np.copyto(best, current)
                if current_f > best_f:
                    best_f = current_f
                    step_last_improved = step_current
                    if current_f > total_best_f:  # new optimum
                        total_best_f = current_f
                        run_last_improved = run_current
                        np.copyto(total_best, current)
                        logger(f"New global best {tuple(best)} with quality"
                               f" {total_best_f} found in run {run_current} "
                               f"after {step_current} FEs.")
        logger(f"Run {run_current} finished with quality {best_f}.")

    result = tuple(total_best)
    logger(f"Finished after {run_current} runs with solution {result} of "
           f"quality {total_best_f}.")
    return result


def __optimize_scales(scale_choices: list[list[tuple[int, int]]],
                      random: Generator) -> list[int]:
    """
    Pick a diverse scale choice.

    :param scale_choices: the scale choices we have per group
    :param random: the random number generator
    :returns: one chosen scale per group
    """
    n: Final[int] = len(scale_choices)

    x_total_best: list[int] = [0] * n
    opt_idx: list[tuple[int, int]] = []  # the relevant indexes
    opt_sum: int = 0
    for idx, choices in enumerate(scale_choices):
        if len(choices) > 1:
            opt_sum += len(choices)
            opt_idx.append((idx, len(choices)))

    if not opt_idx:
        return x_total_best

    def __f(xx: list[int]) -> tuple[float, float, int]:
        """
        Compute the quality of a suggestion, bigger is better.

        :param xx: the candidate solution
        :returns: a tuple with minimum distance, distance sum, and
            difference count
        """
        dist_min: float = inf
        dist_sum: float = 0
        diff_cnt: int = 0
        nonlocal scale_choices
        nonlocal n
        for i in range(n):
            a = scale_choices[i][xx[i]]
            for j in range(i):
                b = scale_choices[j][xx[j]]
                if a[0] == b[0]:
                    d0 = 0
                else:
                    diff_cnt += 1
                    d0 = a[0] - b[0]
                if a[1] == b[1]:
                    d1 = 0
                else:
                    diff_cnt += 1
                    d1 = a[1] - b[1]
                d = sqrt((d0 * d0) + (d1 * d1))
                if d < dist_min:
                    dist_min = d
                dist_sum += d
        return dist_min, dist_sum, diff_cnt

    f_total_best: tuple[float, float, int] = __f(x_total_best)
    logger(f"The following index-choices tuple exist: {opt_idx} and "
           f"the initial choice is {x_total_best} at quality {f_total_best}.")

    x_cur_best: list[int] = [0] * n
    x_cur: list[int] = [0] * n

    for _ in range(int(opt_sum ** 2.35)):
        for ii, sc in enumerate(scale_choices):
            x_cur_best[ii] = random.integers(len(sc))
        f_cur_best: tuple[float, float, int] = __f(x_cur_best)
        if f_cur_best > f_total_best:
            f_total_best = f_cur_best
            x_total_best.clear()
            x_total_best.extend(x_cur_best)
            logger(f"Improved to {x_total_best} at quality {f_total_best}.")
        for _ in range(int(opt_sum ** 2.35)):
            idx, choicen = opt_idx[random.integers(len(opt_idx))]
            old = x_cur[idx]
            while old == x_cur[idx]:
                x_cur[idx] = random.integers(choicen)
            f_cur = __f(x_cur)
            if f_cur >= f_cur_best:
                f_cur_best = f_cur
                x_cur_best.clear()
                x_cur_best.extend(x_cur)
                if f_cur_best > f_total_best:
                    f_total_best = f_cur_best
                    x_total_best.clear()
                    x_total_best.extend(x_cur_best)
                    logger(f"Improved to {x_total_best} "
                           f"at quality {f_total_best}.")

    return x_total_best


def __scale(jobs: int, machines: int) -> int:
    """
    Compute the scale of a JSSP instance.

    :param jobs: the jobs
    :param machines: the machines
    :returns: the scale
    """
    return factorial(jobs) ** machines


def propose_instances(n: int,
                      get_instances: Callable = __get_instances) -> \
        tuple[str, ...]:
    """
    Propose a set of instances to be used for our experiment.

    This function is used to obtain the instances chosen for the JSSP
    experiment. You can also use it for other experiments, by using your own
    instance source and/or for selecting more or less JSSP instances.

    Basically, it will accept a function `get_instances`, which must produce
    a sequence of Instance objects. Each instance has a name (e.g., `dmu14`)
    and a group, where the group is the name without any numerical suffix
    (e.g., `dmu`). This function will then select `n` instances from the
    instance set with the goal to maximize the diversity of the instances, to
    include instances from as many groups as possible, and to include one
    instance of the smallest and one of the largest scale.
    The diversity is measured in terms of the numbers of jobs and machines,
    the instance scale, the minimum and maximum operation length, the
    standard deviation of the mean operation lengths over the jobs, the
    makespan bounds, and so on.

    First, features are computed for each instance. Second, the instances are
    clustered into `n` clusters. Third, we try to pick groups for each cluster
    such that a) the minimum and maximum-scale instances can be included and
    b) that instances from as many groups as possible are picked. Third, we
    then randomly pick one instance for each cluster from the selected group
    (while trying to pick the minimum and maximum-scale instances). Finally,
    the chosen instance names are listed as sorted tuple and returned.

    :param n: the number of instances to be proposed
    :param get_instances: a function returning an
        iterable of instances.
    :return: a tuple with the instance names
    """
    check_int_range(n, "n", 1, 1000)
    if not callable(get_instances):
        raise type_error(get_instances, "get_instances", call=True)
    instances = list(get_instances())
    n_instances: Final[int] = len(instances)
    if n_instances <= 0:
        raise ValueError("No instance returned by get_instance.")
    for i in instances:
        if isinstance(i, Instance):
            raise type_error(
                i, "value in list returned by get_instances", Instance)
    if n_instances <= n:
        if n_instances == n:  # nothing to do here
            return tuple(sorted([inst.get_name() for inst in instances]))
        raise ValueError(f"{n} instances requested, but only "
                         f"{n_instances} available.")
    logger(f"We will pick {n} instances out of {n_instances}.")

    random: Final[Generator] = rand_generator(0)  # the random number generator
    random.shuffle(instances)

    inst_names: Final[tuple[str, ...]] = tuple([  # get the instance names
        inst.get_name() for inst in instances])
    inst_sizes: Final[list[tuple[int, int]]] =\
        [(inst.jobs, inst.machines) for inst in instances]

    # the group to which the instances belong
    rm = str.maketrans("", "", "0123456789")
    inst_groups: Final[tuple[str, ...]] = tuple([  # get the instance groups
        inst.translate(rm) for inst in inst_names])

    # create bi-directional mapping between group names and integer IDs
    group_ids: Final[dict[str, int]] = {}
    id_groups: Final[dict[int, str]] = {}
    for group in inst_groups:
        if group in group_ids:
            continue
        gid = len(group_ids)
        group_ids[group] = gid
        id_groups[gid] = group
    n_groups: Final[int] = len(group_ids)
    logger(f"The {n_instances} instances belong to the {n_groups} "
           f"groups {sorted(group_ids.keys())}.")

    # compute the feature matrix
    base_features: Final[int] = 9
    features = np.zeros((len(instances), base_features + n_groups),
                        DEFAULT_FLOAT)
    min_scale_val = inf
    min_scale_inst: set[int] = set()
    max_scale_val = -inf
    max_scale_inst: set[int] = set()
    for i, inst in enumerate(instances):
        features[i, 0] = inst.jobs
        features[i, 1] = inst.machines
        features[i, 2] = inst.jobs / inst.machines
        scale = __scale(inst.jobs, inst.machines)
        if scale <= min_scale_val:
            if scale < min_scale_val:
                min_scale_val = scale
                min_scale_inst.clear()
            min_scale_inst.add(i)
        if scale >= max_scale_val:
            if scale > max_scale_val:
                max_scale_inst.clear()
                max_scale_val = scale
            max_scale_inst.add(i)
        features[i, 3] = log2(log2(scale))
        m = inst.matrix[:, 1::2]
        features[i, 4] = np.std(np.mean(m, axis=0))
        features[i, 5] = np.min(m)
        features[i, 6] = np.max(m)
        features[i, 7] = inst.makespan_lower_bound
        features[i, 8] = inst.makespan_upper_bound / inst.makespan_lower_bound
        features[i, base_features + group_ids[inst_groups[i]]] = 1
    del instances
    logger(f"We computed {base_features + n_groups} features for each "
           f"instance, namely {base_features} features and {n_groups}"
           f" group features. The instances with the smallest scale "
           f" {min_scale_val} are {[inst_names[i] for i in min_scale_inst]} "
           f" (encoded as {min_scale_inst}) and those of the largest scale "
           f"{max_scale_val} are {[inst_names[i] for i in max_scale_inst]} "
           f"(encoded as {max_scale_inst}). Now we will cluster the "
           f"instances.")

    # standardize feature columns
    for i in range(base_features):
        features[:, i] = (features[:, i] - np.mean(features[:, i])) \
            / np.std(features[:, i])

    # so now we have the following things:
    # 1. a list `instances` of instance names
    # 2. the list `instance_groups` with the corresponding groups
    # 3. set `groups` with the group names
    # 4. the matrix `features` with instance features
    # 5. the bi-directional mapping between instance groups and group IDs
    # 6. the instance/group indexes for the smallest and largest-scale
    #    instances

    # now we cluster the instances
    model = SpectralClustering(n_clusters=n,
                               n_init=100,
                               random_state=RandomState(
                                   random.integers(2 ** 32)))
    clusters = model.fit_predict(features)
    if len(clusters) != n_instances:
        raise ValueError(
            f"Invalid number {len(clusters)} of cluster "
            f"assignments to {n_instances} instances.")
    if (max(clusters) - min(clusters) + 1) != n:
        raise ValueError(f"Expected {n} clusters, but got "
                         f"{max(clusters) - min(clusters) + 1}.")
    logger(f"Found clusters {clusters}. The minimum instances are in"
           f"{[clusters[i] for i in min_scale_inst]}. The maximum instances "
           f"are in {[clusters[i] for i in max_scale_inst]}. now assigning"
           f"clusters to groups.")

    # find which groups belong to which cluster
    cluster_groups: Final[tuple[tuple[int, ...], ...]] = tuple([
        tuple(sorted({group_ids[inst_groups[j]]
                      for j in np.where(clusters == i)[0]}))
        for i in range(n)])
    logger(f"The groups available per cluster are {cluster_groups}.")

    # Now we need to pick the extreme groups.
    extreme_groups = tuple(tuple(sorted(
        {(clusters[xx], group_ids[inst_groups[xx]]) for xx in ex}))
        for ex in [min_scale_inst, max_scale_inst])
    logger(f"The extreme groups are {extreme_groups}.")

    # With this method, we choose one instance group for each cluster.
    chosen_groups = __optimize_clusters(cluster_groups=cluster_groups,
                                        extreme_groups=extreme_groups,
                                        n_groups=n_groups,
                                        random=random)
    logger(f"The instance groups {chosen_groups} were chosen for the"
           f" {n} clusters.")

    # OK, we have picked one instance group per cluster.
    # Now we need to pick the instance scales from these groups.
    # The goal is to now select instances with (jobs, machines) settings as
    # diverse as possible, while adhering to the selected instance groups.
    # For example, a selected group may be of the family "la", but there
    # might be instances of different (job, machines) settings in the cluster
    # for this group.
    # In this case, we want to pick those which do not already occur in other
    # clusters.
    # If we can, we will pick the instances with the minimum and the maximum
    # scales.
    # In a first step, we find out
    needs_min_scale = True
    needs_max_scale = True
    scale_choices: list[list[tuple[int, int]]] = []
    inst_choices: list[list[str]] = []

    for i in range(n):
        elements = np.where(clusters == i)[0]
        sgroup: str = id_groups[chosen_groups[i]]
        possibility: set[int] = {i for i in elements
                                 if inst_groups[i] == sgroup}
        cur_scale_choices: list[tuple[int, int]] = []
        scale_choices.append(cur_scale_choices)
        cur_inst_choices: list[str] = []
        inst_choices.append(cur_inst_choices)
        can_skip: bool = False

        if needs_min_scale:
            test = possibility.intersection(min_scale_inst)
            if len(test) > 0:
                logger(
                    f"Choosing from groups {[inst_names[t] for t in test]} "
                    f"for cluster {i} to facilitate minimum-scale instance.")
                sel = list(test)
                seli = sel[random.integers(len(sel))]
                possibility = {seli}
                cur_scale_choices.append(inst_sizes[seli])
                cur_inst_choices.append(inst_names[seli])
                can_skip = True
                needs_min_scale = False

        if needs_max_scale:
            test = possibility.intersection(max_scale_inst)
            if len(test) > 0:
                logger(
                    f"Choosing from groups {[inst_names[t] for t in test]} "
                    f"for cluster {i} to facilitate maximum-scale instance.")
                needs_max_scale = False
                if can_skip:
                    continue
                sel = list(test)
                seli = sel[random.integers(len(sel))]
                cur_scale_choices.append(inst_sizes[seli])
                cur_inst_choices.append(inst_names[seli])
                continue
        if can_skip:
            continue

        scales: set[tuple[int, int]] = set()
        sel = list(possibility)
        random.shuffle(sel)
        for ii in sel:
            tup = inst_sizes[ii]
            if tup not in scales:
                scales.add(tup)
                cur_inst_choices.append(inst_names[ii])
                cur_scale_choices.append(tup)

    logger(f"Got the scale choices {scale_choices} resulting from the "
           f"possible instances {inst_choices}.")
    # do the actual scale optimization
    final_sel = __optimize_scales(scale_choices, random)

    res_tmp: Final[list[tuple[int, str]]] = \
        [(__scale(scale_choices[i][k][0], scale_choices[i][k][1]),
          inst_choices[i][k]) for i, k in enumerate(final_sel)]
    res_tmp.sort()
    result: Final[list[str]] = [sey[1] for sey in res_tmp]

    # Finally, we sort and finalize the set of chosen instances.
    logger(f"Found final instance selection {result}.")
    return tuple(result)
