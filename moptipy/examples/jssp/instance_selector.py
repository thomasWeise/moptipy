"""Code for selecting interesting instances for smaller-scale experiments."""
from math import factorial, log2, inf
from typing import Dict, Tuple, Final, List, Callable, Set

import numpy as np  # type: ignore
from numpy.random import Generator, RandomState  # type: ignore
from sklearn.cluster import SpectralClustering  # type: ignore

from moptipy.examples.jssp.instance import Instance
from moptipy.utils.nputils import DEFAULT_FLOAT, DEFAULT_INT
from moptipy.utils.nputils import rand_generator
from moptipy.utils.logging import logger


def __get_instances() -> List[Instance]:
    """
    Get the instances.

    :return: a tuple of instances
    :rtype: Tuple[Instance, ...]
    """
    return [Instance.from_resource(name) for name in
            Instance.list_resources() if name != "demo"]


def __optimize_clusters(cluster_groups: Tuple[Tuple[int, ...], ...],
                        n_groups: int,
                        extreme_groups: Tuple[Tuple[Tuple[int,
                                                          int], ...], ...],
                        random: Generator) -> Tuple[int, ...]:
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
    current_f: Tuple[int, int, int, int, float]
    best: Final[np.ndarray] = np.zeros(n, DEFAULT_INT)
    best_f: Tuple[int, int, int, int, float]
    total_best: Final[np.ndarray] = np.zeros(n, DEFAULT_INT)
    total_best_f: Tuple[int, int, int, int, float] = -1, -1, -1, -1, -1
    run_last_improved: int = 1
    run_current: int = 0
    run_max_none_improved: Final[int] = int((32 + (2 * (n * n_groups)))
                                            ** 1.2)
    step_max_none_improved: Final[int] = int(run_max_none_improved ** 1.5)

    done: Final[np.ndarray] = np.zeros(n_groups, DEFAULT_INT)
    extremes: Final[Set[int]] = set()

    logger(f"Beginning to optimize the assignment of {len(cluster_groups)} "
           f"clusters to {n_groups} groups. The minimum groups are "
           f"{extreme_groups[0]}, the maximum groups are {extreme_groups[1]}."
           f" We permit {run_max_none_improved} runs without improvement "
           f"before termination and {step_max_none_improved} FEs without "
           "improvement before stopping a run.")

    def __f(sol: np.ndarray) -> Tuple[int, int, int, int, float]:
        """
        The internal objective function.

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
                if sol[group[0]] == group[1]:
                    if group[0] not in extremes:
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
            if current_f >= best_f:  # accept?
                np.copyto(best, current)
                if current_f > best_f:
                    step_last_improved = step_current
                    if current_f > total_best_f:  # new optimum
                        total_best_f = current_f
                        run_last_improved = run_current
                        np.copyto(total_best, current)
                        logger(f"New global best {tuple(best)} with quality"
                               f" {total_best_f} found in run {run_current} "
                               f"after {step_current} FEs.")
                best_f = current_f

    result = tuple(total_best)
    logger(f"Finished after {run_current} runs with solution {result} of "
           f"quality {total_best_f}.")
    return result


def propose_instances(n: int,
                      get_instances: Callable = __get_instances) -> \
        Tuple[str, ...]:
    """
    Propose a set of instances to be used for our experiment.

    This function was used to obtain `EXPERIMENT_INSTANCES`. For
    `n=len(EXPERIMENT_INSTANCES)`, we now short-circuited it to
    `EXPERIMENT_INSTANCES`.

    :param int n: the number of instances to be proposed
    :param Callable get_instances: a function returning an
        iterable of instances
    :return: a tuple with the instance names
    :rtype: Tuple[str, ...]
    """
    if not (isinstance(n, int)):
        raise TypeError(f"n must be int but is {type(n)}.")
    if n <= 1:
        raise ValueError(f"n must be > 1, but is {n}.")
    if not callable(get_instances):
        raise TypeError("get_instances must be callable, but "
                        f"is {type(get_instances)}.")
    instances = list(get_instances())
    n_instances: Final[int] = len(instances)
    if n_instances <= 0:
        raise ValueError("No instance returned by get_instance.")
    if not all(isinstance(i, Instance) for i in instances):
        raise TypeError("get_instances returned object which "
                        "is not instance of Instance.")
    if n_instances <= n:
        if n_instances == n:  # nothing to do here
            return tuple(sorted([inst.get_name() for inst in instances]))
        raise ValueError(f"{n} instances requested, but only "
                         f"{n_instances} available.")
    logger(f"We will pick {n} instances out of {n_instances}.")

    random: Final[Generator] = rand_generator(0)  # the random number generator
    random.shuffle(instances)

    inst_names: Final[Tuple[str, ...]] = tuple([  # get the instance names
        inst.get_name() for inst in instances])

    # the group to which the instances belong
    rm = str.maketrans("", "", "0123456789")
    inst_groups: Final[Tuple[str, ...]] = tuple([  # get the instance groups
        inst.translate(rm) for inst in inst_names])

    # create bi-directional mapping between group names and integer IDs
    group_ids: Final[Dict[str, int]] = {}
    id_groups: Final[Dict[int, str]] = {}
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
    min_scale_inst: Set[int] = set()
    max_scale_val = -inf
    max_scale_inst: Set[int] = set()
    for i, inst in enumerate(instances):
        features[i, 0] = inst.jobs
        features[i, 1] = inst.machines
        features[i, 2] = inst.jobs / inst.machines
        scale = factorial(inst.jobs) ** inst.machines
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
    cluster_groups: Final[Tuple[Tuple[int, ...], ...]] = tuple([
        tuple(sorted(list({group_ids[inst_groups[j]]
                           for j in np.where(clusters == i)[0]})))
        for i in range(n)])
    logger(f"The groups available per cluster are {cluster_groups}.")

    # Now we need to pick the extreme groups.
    extreme_groups = tuple(tuple(sorted(list(
        {(clusters[xx], group_ids[inst_groups[xx]]) for xx in ex})))
        for ex in [min_scale_inst, max_scale_inst])
    logger(f"The extreme groups are {extreme_groups}.")

    # With this method, we choose one instance group for each cluster.
    chosen_groups = __optimize_clusters(cluster_groups=cluster_groups,
                                        extreme_groups=extreme_groups,
                                        n_groups=n_groups,
                                        random=random)
    logger(f"The instance groups {chosen_groups} were chosen for the"
           f" {n} clusters.")
    result: List[str] = []

    # OK, we have picked one instance group per cluster.
    # Now we pick the instance from that group.
    # If we can, we will pick the instances with the minimum and the maximum
    # scales.
    needs_min_scale = True
    needs_max_scale = True
    for i in range(n):
        elements = np.where(clusters == i)[0]
        sgroup: str = id_groups[chosen_groups[i]]
        possibility: Set[int] = {i for i in elements
                                 if inst_groups[i] == sgroup}
        if needs_min_scale:
            test = possibility.intersection(min_scale_inst)
            if len(test) > 0:
                logger(
                    f"Choosing from groups {[inst_names[t] for t in test]} "
                    f"for cluster {i} to facilitate minimum-scale instance.")
                possibility = test
                needs_min_scale = False
        if needs_max_scale:
            test = possibility.intersection(max_scale_inst)
            if len(test) > 0:
                logger(
                    f"Choosing from groups {[inst_names[t] for t in test]} "
                    f"for cluster {i} to facilitate maximum-scale instance.")
                possibility = test
                needs_max_scale = False
        sel = list(possibility)
        result.append(inst_names[sel[random.integers(len(sel))]])

    # Finally, we sort and finalize the set of chosen instances.
    result.sort()
    logger(f"Found final instance selection {result}.")
    return tuple(result)
