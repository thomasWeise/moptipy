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
from math import factorial
from typing import Tuple

import numba  # type: ignore
import numpy as np


def gantt_space_size(jobs: int, machines: int) -> int:
    """
    Compute the size of the Gantt space.

    :param int jobs: the number of jobs
    :param int machines: the number of machines
    :return: the size of the search
    :rtype: int
    """
    if not isinstance(jobs, int):
        raise TypeError(f"Number of jobs must be int, but is {type(jobs)}.")
    if jobs <= 0:
        raise ValueError(f"Number of jobs must be > 0, but is {jobs}.")
    if not isinstance(machines, int):
        raise TypeError(
            f"Number of machines must be int, but is {type(machines)}.")
    if machines <= 0:
        raise ValueError(
            f"Number of machines must be > 0, but is {machines}.")
    return factorial(jobs) ** machines


#: the pre-computed values
__PRE_COMPUTED: Tuple[Tuple[int, int, int,
                            Tuple[Tuple[int, ...], ...]], ...] = (
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
     ((0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0)))
)


def gantt_min_feasible(jobs: int, machines: int) \
        -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
    """
    Find the minimum number of feasible gantt charts.

    :param int jobs: the number of jobs
    :param int machines: the number of machines
    :return: the minimum number of feasible solutions for any instance
        of the given configuration and one example of such an instance
    :rtype: Tuple[int, Tuple[Tuple[int, ...], ...]]
    """
    if not isinstance(jobs, int):
        raise TypeError(f"Number of jobs must be int, but is {type(jobs)}.")
    if (jobs <= 0) or (jobs > 127):
        raise ValueError(f"Number of jobs must be in 1..127, but is {jobs}.")
    if not isinstance(machines, int):
        raise TypeError(
            f"Number of machines must be int, but is {type(machines)}.")
    if (machines <= 0) or (machines > 127):
        raise ValueError(
            f"Number of machines must be n 1..127, but is {machines}.")

    if machines <= 1:
        return factorial(jobs), tuple([tuple([0] * jobs)])
    if jobs <= 1:
        return 1, tuple([tuple(range(machines))])
    if jobs <= 2:
        return machines + 1, tuple([
            tuple(list(range(machines - 2, -1, -1)) + [machines - 1]),
            tuple([machines - 1] + list(range(machines - 1)))])

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

    :param np.ndarray dest: the destination
    :param np.ndarray source: the source
    :param np.int64 n: the number of elements to copy
    """
    for a in range(n):
        dest[a] = source[a]


@numba.njit
def __copy_instance(dest: np.ndarray, source: np.ndarray,
                    jobs: np.int64, machines: np.int64) -> None:
    """
    Copy a instance.

    :param np.ndarray dest: the destination
    :param np.ndarray source: the source
    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the machines
    """
    for a in range(jobs):
        __copy(dest[a], source[a], machines)


@numba.njit
def __find_min_feasible(jobs: np.int64, machines: np.int64,
                        dest: np.ndarray) -> np.int64:
    """
    Find the minimum number of feasible gantt charts.

    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the number of machines
    :param np.ndarray dest: the destination array
    :return: the minimum number of feasible solutions for any instance
        of the given configuration
    :rtype: np.int64
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

    :param np.ndarray instance: the instance
    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the number of machines
    :return: True if the instance is sorted, False otherwise
    :rtype: bool
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

    :param np.ndarray instance: the JSSP instance, size jobs*machines
    :param np.ndarray gantt: the gantt chart, size machines*jobs
    :param np.ndarray job_state: the job state, of length jobs
    :param np.ndarray gantt_state: the machine state, of length machines
    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the number of machines
    :param np.int64 row: the number of valid rows of the Gantt chart
    :return: True if the chart is feasible so far, False otherwise
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

    :param np.ndarray arr: the array to permute over
    :param np.ndarray index: the array with the index
    :param np.int64 pi: the index of the index to use in pi
    :param np.int64 n: the length of arr
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

    :param np.ndarray arr: the array to permute over
    :param np.ndarray index: the array with the index
    :param int pi: the index of the index to use in pi
    :param int n: the length of arr
    :returns: True if there is a next permutation, False if not
    :rtype: bool
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
            # binary search to find the greatest value which is less than
            # arr[idx + 1]
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

    :param np.ndarray instance: the JSSP instance, size jobs*machines
    :param np.ndarray gantt: the gantt chart array, size machines*jobs
    :param np.ndarray job_state: the job state, of length jobs
    :param np.ndarray gantt_state: the machine state, of length machines
    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the number of machines
    :param np.int64 upper_bound: the upper bound - we won't enumerate more
        charts than this.
    :param np.ndarray index: the index array
    :param np.ndarray dest: the destination array
    :returns: the number of enumerated feasible gantt charts
    :rtype: np.int64
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

    :param np.int64 jobs: the number of jobs
    :param np.int64 machines: the number of machines
    :param np.ndarray instance: the provided instance array
    :return: the minimum number of feasible solutions for any instance
        of the given configuration
    :rtype: np.int64
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
