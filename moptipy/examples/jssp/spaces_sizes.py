"""
Computations regarding the size of the solution spaces in a JSSP.

We represent solutions for a Job Shop Scheduling Problems (JSSPs) as Gantt
diagrams.
Assume that we look at JSSPs with n jobs and m machines.
The number of possible Gantt charts (without useless delays) is
(jobs!)**machines.

However, not all of them are necessarily feasible.
If all jobs pass through all machines in the same order, then all of the
possible Gantt charts are also feasible.
However, if, say job 0 first goes to machine 0 and then to machine 1 and
job 1 first goes to machine 1 and then to machine 0, there are possible
Gantt charts with deadlocks: If we put the second operation of job 0 to
be the first operation to be done by machine 1 and put the second operation
of job 1 to be the first operation to be done by machine 0, we end up with
an infeasible Gantt chart, i.e., one that cannot be executed.
Thus, the question arises: "For a given number n of jobs and m of machines,
what is the instance with the fewest feasible Gantt charts?"

Well, I am sure that there are clever algorithms to compute this. Maybe we
can even ave an elegant combinatorial formula.
But I do not want to spend much time on this subject (and maybe would not
be able to figure it out even if I wanted to...).
So we try to find this information using a somewhat brute force approach:
By enumerating instances and, for each instance, the Gantt charts, and
count how many of them are feasible.
"""
import sys
from math import factorial, log10
from typing import Iterable

import numba  # type: ignore
import numpy as np

from moptipy.examples.jssp import experiment
from moptipy.examples.jssp.gantt_space import gantt_space_size
from moptipy.examples.jssp.instance import Instance
from moptipy.utils.lang import Lang
from moptipy.utils.path import Path
from moptipy.utils.types import check_int_range


def permutations_with_repetitions_space_size(n: int, m: int) -> int:
    """
    Compute the number of n-permutations with m repetitions.

    :param n: the number of different values
    :param m: the number of repetitions
    :returns: the space size
    :rtype: int
    """
    return factorial(n * m) // (factorial(m) ** n)


#: the pre-computed values
__PRE_COMPUTED: tuple[tuple[int, int, int,
                            tuple[tuple[int, ...], ...]], ...] = (
    (3, 2, 22, ((0, 1), (0, 1), (1, 0))),
    (3, 3, 63, ((0, 1, 2), (1, 0, 2), (2, 0, 1))),
    (3, 4, 147, ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1))),
    (3, 5, 317, ((0, 1, 2, 3, 4), (2, 1, 0, 4, 3), (3, 4, 0, 2, 1))),
    (4, 2, 244, ((0, 1), (0, 1), (1, 0), (1, 0))),
    (4, 3, 1630, ((0, 1, 2), (1, 0, 2), (2, 0, 1), (2, 1, 0))),
    (4, 4, 7451, ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 1, 0), (3, 2, 0, 1))),
    (5, 2, 4548, ((0, 1), (0, 1), (0, 1), (1, 0), (1, 0))),
    (5, 3, 91461, ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1))),
    (6, 2, 108828, ((0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0))),
    (7, 2, 3771792, ((0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0))),
    (8, 2, 156073536,
     ((0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0))),
)


def gantt_min_feasible(jobs: int, machines: int) \
        -> tuple[int, tuple[tuple[int, ...], ...]]:
    """
    Find the minimum number of feasible gantt charts.

    :param jobs: the number of jobs
    :param machines: the number of machines
    :return: the minimum number of feasible solutions for any instance
        of the given configuration and one example of such an instance
    """
    check_int_range(jobs, "jobs", 1, 127)
    check_int_range(machines, "machines", 1, 127)

    if machines <= 1:
        return factorial(jobs), (tuple([0] * jobs), )
    if jobs <= 1:
        return 1, (tuple(range(machines)), )
    if jobs <= 2:
        return machines + 1, (
            (*list(range(machines - 2, -1, -1)), machines - 1),
            (machines - 1, *list(range(machines - 1))))

    for tup in __PRE_COMPUTED:
        if (tup[0] == jobs) and (tup[1] == machines):
            return tup[2], tup[3]

    if machines <= 2:  # if there are two machines, we know the shape
        lst = [[0, 1]] * (jobs - (jobs // 2))
        lst.extend([[1, 0]] * (jobs // 2))
        dest = np.array(lst, dtype=np.uint8)
        res = __enumerate_feasible_for(jobs, machines, dest)
    else:  # more than two machines: need to enumerate
        dest = np.ndarray(shape=(jobs, machines), dtype=np.uint8)
        res = int(__find_min_feasible(np.int64(jobs),
                                      np.int64(machines), dest))

    # turn the result into a tuple
    arr = tuple(sorted([tuple([int(dest[i, j]) for j in range(machines)])
                        for i in range(jobs)]))
    return res, arr


@numba.njit
def __copy(dest: np.ndarray, source: np.ndarray, n: np.int64) -> None:
    """
    Copy an array.

    :param dest: the destination
    :param source: the source
    :param n: the number of elements to copy
    """
    for a in range(n):
        dest[a] = source[a]


@numba.njit
def __copy_instance(dest: np.ndarray, source: np.ndarray,
                    jobs: np.int64, machines: np.int64) -> None:
    """
    Copy an instance.

    :param dest: the destination
    :param source: the source
    :param jobs: the number of jobs
    :param machines: the machines
    """
    for a in range(jobs):
        __copy(dest[a], source[a], machines)


@numba.njit
def __find_min_feasible(jobs: np.int64, machines: np.int64,
                        dest: np.ndarray) -> np.int64:
    """
    Find the minimum number of feasible gantt charts.

    :param jobs: the number of jobs
    :param machines: the number of machines
    :param dest: the destination array
    :return: the minimum number of feasible solutions for any instance
        of the given configuration
    """
    instance = np.empty(shape=(jobs, machines), dtype=np.uint8)
    gantt = np.empty(shape=(machines, jobs), dtype=np.uint8)
    gantt_index = np.zeros(machines, dtype=np.int64)
    inst_index = np.zeros(jobs, dtype=np.int64)
    job_state = np.zeros(jobs, dtype=np.int64)
    gantt_state = np.zeros(machines, dtype=np.int64)
    upper_bound: np.int64 = np.int64(9223372036854775807)

    for i in range(jobs):
        __first_perm(instance[i], inst_index, i, machines)

    while True:
        if __check_sorted(instance, jobs, machines):
            upper_bound = __enumerate_feasible(
                instance, gantt, job_state, gantt_state, jobs, machines,
                upper_bound, gantt_index, dest)

        k = jobs - 1
        while True:
            if __next_perm(instance[k], inst_index, k, machines):
                break
            k = k - 1
            if k < 0:
                return upper_bound
        for j in range(k + 1, jobs):
            __first_perm(instance[j], inst_index, j, machines)


@numba.njit(nogil=True)
def __check_sorted(instance: np.ndarray,
                   jobs: np.int64,
                   machines: np.int64) -> bool:
    """
    Check if the instance is such that all jobs are sorted.

    :param instance: the instance
    :param jobs: the number of jobs
    :param machines: the number of machines
    :return: `True` if the instance is sorted, `False` otherwise
    """
    i: np.int64 = jobs - 1
    arr1: np.ndarray = instance[i]
    while i > 0:
        arr2 = arr1
        i = i - 1
        arr1 = instance[i]
        for j in range(machines):
            if arr1[j] > arr2[j]:
                return True
            if arr1[j] < arr2[j]:
                return False
    return True


@numba.njit
def __is_feasible(instance: np.ndarray,
                  gantt: np.ndarray,
                  job_state: np.ndarray,
                  gantt_state: np.ndarray,
                  jobs: np.int64,
                  machines: np.int64,
                  row: np.int64) -> bool:
    """
    Check if a Gantt diagram populated until a given row is feasible.

    :param instance: the JSSP instance, size jobs*machines
    :param gantt: the gantt chart, size machines*jobs
    :param job_state: the job state, of length jobs
    :param gantt_state: the machine state, of length machines
    :param jobs: the number of jobs
    :param machines: the number of machines
    :param row: the number of valid rows of the Gantt chart
    :return: `True` if the chart is feasible so far, `False` otherwise
    """
    if row <= 1:
        return True
    job_state.fill(0)  # all jobs start at the 0'th operations
    gantt_state.fill(0)  # all machines start at op 0
    found: bool = True
    needed_jobs = jobs  # the number of required jobs

    while found:
        found = False
        for job in range(jobs):  # check all jobs
            while True:  # we process each job as long as we can
                js = job_state[job]  # which operation is required?
                if js >= machines:
                    break  # the job is already finished
                nm = instance[job, js]  # the machine that the operation needs
                if nm >= row:  # ok, this machine is outside of the chart
                    js += 1  # so we assume it's ok
                    job_state[job] = js  # and step forward the state
                    found = True  # we did something!
                    if js >= machines:  # oh, we finished the job?
                        needed_jobs -= 1  # one less jobs to do
                        if needed_jobs <= 0:  # no more jobs to do?
                            return True  # the chart is feasible!
                        break  # quit handling this job, as its finished
                    continue  # move to next operation, as job is not finished
                ms = gantt_state[nm]  # get next item in gantt chart
                mj = gantt[nm, ms]  # next job on this machine?
                if mj == job:  # great, this job!
                    gantt_state[nm] = ms + 1  # advance gantt state
                    js += 1  # so we can perform the operation
                    job_state[job] = js  # and step forward the state
                    found = True  # we did something!
                    if js >= machines:  # oh, we finished the job?
                        needed_jobs -= 1  # one less jobs to do
                        if needed_jobs <= 0:  # no more jobs to do?
                            return True  # the chart is feasible!
                        break  # quit handling this job, as its finished
                    continue  # move to next operation, as job is not finished
                break  # no, we cannot handle this job now

    return False  # we ended one round without being able to proceed


@numba.njit
def __first_perm(arr: np.ndarray,
                 index: np.ndarray,
                 pi: np.int64,
                 n: np.int64) -> None:
    """
    Create the first permutation for a given array of values.

    :param arr: the array to permute over
    :param index: the array with the index
    :param pi: the index of the index to use in pi
    :param n: the length of arr
    """
    for i in range(n):
        arr[i] = i
    index[pi] = np.int64(0)


@numba.njit
def __next_perm(arr: np.ndarray,
                index: np.ndarray,
                pi: np.int64,
                n: np.int64) -> bool:
    """
    Get the next permutation for a given array of values.

    :param arr: the array to permute over
    :param index: the array with the index
    :param pi: the index of the index to use in pi
    :param n: the length of arr
    :returns: `True` if there is a next permutation, `False` if not
    """
    idx = index[pi]
    if idx >= n - 1:
        return False

    nidx = idx + 1

    if idx == 0:  # increase is at the very beginning
        arr[0], arr[1] = arr[1], arr[0]  # swap
        idx = nidx
        while True:  # update index to the next increase
            nidx = idx + 1
            if nidx >= n:
                break  # reached end
            if arr[idx] <= arr[nidx]:
                break  # found increase
            idx = nidx
    else:
        if arr[nidx] > arr[0]:  # value at arr[idx + 1] is greater than arr[0]
            # no need for binary search, just swap arr[idx + 1] and arr[0]
            arr[nidx], arr[0] = arr[0], arr[nidx]
        else:
            # binary search to find the greatest value which is less
            # than arr[idx + 1]
            start = np.int64(0)
            end = idx
            mid = (start + end) // 2
            t_value = arr[nidx]
            while not (arr[mid] < t_value < arr[mid - 1]):
                if arr[mid] < t_value:
                    end = mid - 1
                else:
                    start = mid + 1
                mid = (start + end) // 2
            arr[nidx], arr[mid] = arr[mid], arr[nidx]  # swap

        # invert 0 to increase
        for i in range((idx // 2) + 1):
            arr[i], arr[idx - i] = arr[idx - i], arr[i]
        idx = 0  # reset increase

    index[pi] = idx
    return True


@numba.njit
def __enumerate_feasible(instance: np.ndarray,
                         gantt: np.ndarray,
                         job_state: np.ndarray,
                         gantt_state: np.ndarray,
                         jobs: np.int64,
                         machines: np.int64,
                         upper_bound: np.int64,
                         index: np.ndarray,
                         dest: np.ndarray) -> np.int64:
    """
    Enumerate the feasible gantt charts for an instance.

    :param instance: the JSSP instance, size jobs*machines
    :param gantt: the gantt chart array, size machines*jobs
    :param job_state: the job state, of length jobs
    :param gantt_state: the machine state, of length machines
    :param jobs: the number of jobs
    :param machines: the number of machines
    :param upper_bound: the upper bound - we won't enumerate more
        charts than this.
    :param index: the index array
    :param dest: the destination array
    :returns: the number of enumerated feasible gantt charts
    """
    counter: np.int64 = np.int64(0)
    for z in range(machines):
        __first_perm(gantt[z], index, z, jobs)

    while True:
        if __is_feasible(instance, gantt, job_state, gantt_state,
                         jobs, machines, machines):
            counter += 1  # found another feasible gantt chart for instance
            if counter >= upper_bound:
                return counter  # if we have reached the minimum, we can stop

        i = machines - 1
        while True:
            if __next_perm(gantt[i], index, i, jobs):
                if i >= machines - 1:
                    break
                if __is_feasible(instance, gantt, job_state, gantt_state,
                                 jobs, machines, i + 1):
                    for j in range(i + 1, machines):
                        __first_perm(gantt[j], index, j, jobs)
                    break
            else:
                i -= 1
                if i < 0:
                    if instance is not dest:
                        __copy_instance(dest, instance, jobs, machines)
                    return counter  # we have enumerated all gantt charts


@numba.njit
def __enumerate_feasible_for(jobs: np.int64, machines: np.int64,
                             instance: np.ndarray) -> np.int64:
    """
    Find the minimum number of feasible gantt charts.

    :param jobs: the number of jobs
    :param machines: the number of machines
    :param instance: the provided instance array
    :return: the minimum number of feasible solutions for any instance
        of the given configuration
    """
    gantt = np.empty(shape=(machines, jobs), dtype=np.uint8)
    gantt_index = np.zeros(machines, dtype=np.int64)
    job_state = np.zeros(jobs, dtype=np.int64)
    gantt_state = np.zeros(machines, dtype=np.int64)
    upper_bound: np.int64 = np.int64(9223372036854775807)
    return __enumerate_feasible(instance, gantt, job_state,
                                gantt_state, jobs, machines,
                                upper_bound, gantt_index,
                                instance)


def __long_str(value: int) -> str:
    """
    Convert a value to a string.

    :param value: the value
    :returns: the string representation
    """
    if value < 0:
        return ""
    if value <= 1_000_000_000_000:
        return Lang.current().format_int(value)
    logg = log10(value)
    exp = int(logg)
    base = value / (10 ** exp)
    expf = Lang.current().format_int(exp)
    return f"$\\approx$&nbsp;{base:.3f}*10^{expf}^"


def make_gantt_space_size_table(
        dest: str = "solution_space_size.md",
        instances: Iterable[str] = tuple(list(  # noqa
            experiment.INSTANCES) + ["demo"])) -> Path:  # noqa
    """
    Print a table of solution space sizes.

    :param dest: the destination file
    :param instances: the instances to add
    :returns: the fully-qualified path to the generated file
    """
    file = Path.path(dest)
    text = [(f'|{Lang.current()["name"]}|'
             r"$\jsspJobs$|$\jsspMachines$|$\min(\#\text{"
             f'{Lang.current()["feasible"]}'
             r"})$|$\left|\solutionSpace\right|$|"),
            r"|:--|--:|--:|--:|--:|"]

    inst_scales: list[tuple[int, int, int, int, str]] = []

    # enumerate the pre-defined instances
    for inst in set(instances):
        instance = Instance.from_resource(inst)
        min_size = -1
        for tup in __PRE_COMPUTED:
            if (tup[0] == instance.jobs) and (tup[1] == instance.machines):
                min_size = tup[2]
                break
        if (min_size < 0) and ((instance.jobs <= 2)
                               or (instance.machines <= 2)):
            min_size = gantt_min_feasible(
                instance.jobs, instance.machines)[0]
        inst_scales.append(
            (instance.jobs, instance.machines,
             gantt_space_size(instance.jobs, instance.machines),
             min_size, f"`{instance.name}`"))
        del instance

    # enumerate some default values
    for jobs in range(2, 6):
        for machines in range(2, 6):
            found: bool = False  # skip over already added scales
            for tupp in inst_scales:
                if (tupp[0] == jobs) and (tupp[1] == machines):
                    found = True
                    break
            if found:
                continue

            min_size = -1
            for tup in __PRE_COMPUTED:
                if (tup[0] == jobs) and (tup[1] == machines):
                    min_size = tup[2]
                    break
            if (min_size < 0) and ((jobs <= 2) or (machines <= 2)):
                min_size = gantt_min_feasible(jobs, machines)[0]
            name = "[@fig:jssp_feasible_gantt]" \
                if (jobs == 2) and (machines == 2) else ""
            inst_scales.append(
                (jobs, machines, gantt_space_size(jobs, machines),
                 min_size, name))

    inst_scales.sort()
    for i, ua in enumerate(inst_scales):
        a = ua
        for j in range(i, len(inst_scales)):
            b = inst_scales[j]
            if (a[-1] and b[-1]) and (a[-3] > b[-3]):
                inst_scales[i] = b
                inst_scales[j] = a
                a = b

    for scale in inst_scales:
        text.append(f"|{scale[4]}|{scale[0]}|{scale[1]}|"
                    f"{__long_str(scale[3])}|{__long_str(scale[2])}|")

    file.write_all(text)
    file.enforce_file()
    return file


def make_search_space_size_table(
        dest: str = "solution_space_size.md",
        instances: Iterable[str] = tuple(list(  # noqa
            experiment.INSTANCES) + ["demo"])) -> Path:  # noqa
    """
    Print a table of search space sizes.

    :param dest: the destination file
    :param instances: the instances to add
    :returns: the fully-qualified path to the generated file
    """
    file = Path.path(dest)
    text = [(f'|{Lang.current()["name"]}|'
             r"$\jsspJobs$|$\jsspMachines$|$\left|\solutionSpace\right|$|"
             r"$\left|\searchSpace\right|$|"),
            r"|:--|--:|--:|--:|--:|"]
    inst_scales: list[tuple[int, int, int, int, str]] = []

    # enumerate the pre-defined instances
    for inst in set(instances):
        instance = Instance.from_resource(inst)
        inst_scales.append(
            (instance.jobs, instance.machines,
             gantt_space_size(instance.jobs, instance.machines),
             permutations_with_repetitions_space_size(
                 instance.jobs, instance.machines),
             f"`{instance.name}`"))
        del instance

    # enumerate some default values
    for jobs in range(3, 6):
        for machines in range(2, 6):
            found: bool = False  # skip over already added scales
            for tupp in inst_scales:
                if (tupp[0] == jobs) and (tupp[1] == machines):
                    found = True
                    break
            if found:
                continue
            inst_scales.append(
                (jobs, machines, gantt_space_size(jobs, machines),
                 permutations_with_repetitions_space_size(
                     jobs, machines), ""))

    inst_scales.sort()
    for i, ua in enumerate(inst_scales):
        a = ua
        for j in range(i, len(inst_scales)):
            b = inst_scales[j]
            if (a[-1] and b[-1]) and (a[-2] > b[-2]):
                inst_scales[i] = b
                inst_scales[j] = a
                a = b
    for scale in inst_scales:
        text.append(f"|{scale[4]}|{scale[0]}|{scale[1]}|"
                    f"{__long_str(scale[2])}|{__long_str(scale[3])}|")

    file.write_all(text)
    file.enforce_file()
    return file


# create the tables if this is the main script
if __name__ == "__main__":
    dest_dir = Path.path(sys.argv[1])
    dest_dir.ensure_dir_exists()
    for lang in Lang.all_langs():
        lang.set_current()
        make_gantt_space_size_table(
            dest_dir.resolve_inside(
                lang.filename("solution_space_size") + ".md"))
        make_search_space_size_table(
            dest_dir.resolve_inside(
                lang.filename("search_space_size") + ".md"))
