"""Test the frequency computation."""

from collections import Counter
from os.path import dirname
from typing import Final, cast

from numpy.random import Generator, default_rng
from pycommons.io.path import Path
from pycommons.io.temp import temp_dir, temp_file

from moptipy.evaluation.base import PerRunData
from moptipy.evaluation.frequency import (
    from_logs,
    number_of_objective_values_to_csv,
)


class Example:
    """An example for the problem."""

    def __init__(self,
                 algorithm: str = "fea_revn",
                 instance: str = "a280",
                 seed: int | None = None,
                 random: Generator = default_rng()) -> None:
        """
        Create the example log file dataset.

        :param algorithm: the algorithm name to use
        :param instance: the instance name
        :param seed: the seed to use
        :param random: the random number generator.
        """
        self.lb: Final[int] = int(random.integers(0, 100000))
        self.goal: Final[int] = self.lb + int(random.integers(100, 10000))
        assert self.goal > self.lb
        self.best: Final[int] = self.goal + int(random.integers(100, 10000))
        assert self.best > self.goal
        self.worst: Final[int] = self.best + int(
            random.integers(10000, 1000000))
        assert self.worst >= self.best + 10000
        self.ub: Final[int] = self.worst + int(random.integers(100, 10000))
        assert self.ub > self.worst

        self.progress_h: Final[dict[int, int]] = {
            self.best: int(random.integers(1, 100)),
            self.worst: int(random.integers(1, 100))}
        n: int = int(random.integers(len(self.progress_h) + 1, 50))
        while len(self.progress_h) < n:
            self.progress_h[int(random.uniform(self.best, self.worst))] \
                = int(random.integers(1, 100))
        self.progress: Final[tuple[int, ...]] = tuple(sorted(
            self.progress_h.keys(), reverse=True))
        assert len(self.progress_h) == n
        assert self.best in self.progress_h
        assert self.worst in self.progress_h
        assert all(self.best <= k <= self.worst for k in self.progress_h)

        self.all_h: Final[dict[int, int]] = dict(self.progress_h)
        n += int(random.integers(1, 10))
        while len(self.all_h) < n:
            self.all_h[int(random.uniform(self.best, self.worst))] \
                = int(random.integers(1, 100))
        assert len(self.all_h) == n
        assert self.best in self.all_h
        assert self.worst in self.all_h
        assert all(self.best <= k <= self.worst for k in self.all_h)
        assert len(self.progress_h) < len(self.all_h)

        self.seed: Final[int] = (
            int(random.integers(10000000))) if seed is None else seed
        self.algorithm: Final[str] = algorithm
        self.instance: Final[str] = instance

        data: list[str] = ["BEGIN_PROGRESS\nfes;timeMS;f"]
        fe: int = 1
        time: int = int(random.integers(1, 10))
        for prg in self.progress:
            time += int(random.integers(1, 10))
            data.append(f"{fe};{time};{prg}")
            time += int(random.integers(1, 10))
            fe += int(random.integers(1, 100))
        data.append("END_PROGRESS\nBEGIN_STATE")
        data.append(f"totalFEs: {fe + random.integers(1, 100)}")
        data.append(f"totalTimeMillis: {time + random.integers(1, 100)}")
        data.append(f"bestF: {self.best}")
        data.append(f"lastImprovementFE: {fe}")
        data.append(f"lastImprovementTimeMillis: {time}")
        data.append("END_STATE\nBEGIN_SETUP")
        data.append(f"a.name: {self.algorithm}")
        data.append(f"p.goalF: {self.goal}")
        data.append(f"p.randSeed: {self.seed}")
        data.append(f"f.lowerBound: {self.lb}")
        data.append(f"f.upperBound: {self.ub}")
        data.append("f.name: tourLength\nEND_SETUP\nBEGIN_SYS_INFO")
        data.append("os.name: Linux\nEND_SYS_INFO\nBEGIN_RESULT_Y")
        data.append("1;2;3\nEND_RESULT_Y\nBEGIN_H")
        data.append(";".join(f"{k};{v}" for k, v in self.all_h.items()))
        data.append("END_H")
        self.file_contents: Final[str] = "\n".join(s.strip() for s in data)

    def log_name(self) -> str:
        """Get the proper name for a log file."""
        return (f"{self.algorithm}/{self.instance}/"
                f"{self.algorithm}_{self.instance}_{hex(self.seed)}.txt")


def test_frequency() -> None:
    """Test the frequency evaluation."""
    with temp_dir() as td:
        exa: Final[Example] = Example()
        file = td.resolve_inside(exa.log_name())
        Path(dirname(file)).ensure_dir_exists()
        file.write_all_str(exa.file_contents)

        data: list = []

        def __append(a: PerRunData, b: Counter) -> None:
            nonlocal data
            assert isinstance(a, PerRunData)
            assert isinstance(b, Counter)
            data.append(a)
            data.append(Counter(b))

        from_logs(td, __append,
                  report_progress=False,
                  report_lower_bound=False,
                  report_upper_bound=False,
                  report_goal_f=False,
                  report_h=False)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        assert len(value) == 1
        assert next(iter(value.keys())) == exa.best
        assert value[exa.best] == 1
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=False,
                  report_upper_bound=False,
                  report_goal_f=False,
                  report_h=False)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        keys = tuple(sorted(value.keys(), reverse=True))
        assert keys == exa.progress
        assert all(value[k] == 1 for k in keys)
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=True,
                  report_upper_bound=False,
                  report_goal_f=False,
                  report_h=False)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        kk = list(exa.progress)
        kk.append(exa.lb)
        kk.sort()
        keys = sorted(value.keys())
        assert keys == kk
        assert all(value[k] == 1 for k in keys)
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=True,
                  report_upper_bound=True,
                  report_goal_f=False,
                  report_h=False)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        kk = list(exa.progress)
        kk.append(exa.lb)
        kk.append(exa.ub)
        kk.sort()
        keys = sorted(value.keys())
        assert keys == kk
        assert all(value[k] == 1 for k in keys)
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=True,
                  report_upper_bound=True,
                  report_goal_f=True,
                  report_h=False)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        kk = list(exa.progress)
        kk.append(exa.lb)
        kk.append(exa.ub)
        kk.append(exa.goal)
        kk.sort()
        keys = sorted(value.keys())
        assert keys == kk
        assert all(value[k] == 1 for k in keys)
        data.clear()

        from_logs(td, __append,
                  report_progress=False,
                  report_lower_bound=False,
                  report_upper_bound=False,
                  report_goal_f=False,
                  report_h=True)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        assert value == exa.all_h
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=False,
                  report_upper_bound=False,
                  report_goal_f=False,
                  report_h=True)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        expc: dict[int, int] = Counter(exa.all_h) + Counter(exa.progress)
        assert value == expc
        data.clear()

        from_logs(td, __append,
                  report_progress=True,
                  report_lower_bound=True,
                  report_upper_bound=True,
                  report_goal_f=True,
                  report_h=True)
        assert len(data) == 2
        key = cast(PerRunData, data[0])
        value = cast(Counter[int | float], data[1])
        assert key.instance == exa.instance
        assert key.algorithm == exa.algorithm
        assert key.encoding is None
        assert key.rand_seed == exa.seed
        assert key.objective == "tourLength"
        expc: dict[int, int] = Counter(exa.all_h) + Counter(exa.progress)
        expc[exa.lb] = 1
        expc[exa.ub] = 1
        expc[exa.goal] = 1
        assert value == expc
        data.clear()


def test_number_of_objective_values_to_csv() -> None:
    """Test the number of objective values computation."""
    instances: tuple[str, ...] = ("i1", "i2", "i3")
    algorithms: tuple[str, ...] = ("a1", "a2", "a3")
    inst_count: dict[str, int] = {}
    algo_inst_count: dict[tuple[str, str], int] = {}
    runs: list[Example] = []
    random: Generator = default_rng()
    for inst in instances:
        inst_ob: set[int] = set()
        for algo in algorithms:
            first: bool = True
            algo_ob: set[int] = set()
            while first or random.integers(5) > 0:
                first = False
                exa = Example(algo, inst, 100 * len(runs))
                runs.append(exa)
                inst_ob.update(exa.all_h.keys())
                algo_ob.update(exa.all_h.keys())
            algo_inst_count[(algo, inst)] = len(algo_ob)
        inst_count[inst] = len(inst_ob)

    with temp_dir() as td:
        for exa in runs:
            path = td.resolve_inside(exa.log_name())
            Path(dirname(path)).ensure_dir_exists()
            path.write_all_str(exa.file_contents)
        with temp_file() as tf:
            number_of_objective_values_to_csv(td, tf)
            lines = tf.read_all_str().splitlines()

    expected: list[str] = [
        "instance;" + ";".join(a for a in algorithms) + ";all"]
    for inst in instances:
        line = inst
        for algo in algorithms:
            line += ";" + str(algo_inst_count[(algo, inst)])
        line += ";" + str(inst_count[inst])
        expected.append(line)

    assert lines == expected
