"""Test the execution of an experiment and parsing the log files the JSSP."""
from typing import Final

import numpy as np

import moptipy.evaluation.progress as prg
from moptipy.evaluation.progress import Progress
from moptipy.evaluation.stat_run import StatRun


def test_stat_runs() -> None:
    """Test whether StatRuns work."""
    algo0: Final[str] = "a0"
    inst0: Final[str] = "i0"
    tu: Final[str] = prg.TIME_UNIT_FES
    fn: Final[str] = prg.F_NAME_RAW
    p0: Final[Progress] = Progress(algo0,
                                   inst0,
                                   0,
                                   np.array([10, 20, 30, 40, 50, 60, 70]),
                                   tu,
                                   np.array([100, 90, 80, 70, 60, 50, 40]),
                                   fn,
                                   None,
                                   True)

    collector: Final[list[StatRun]] = []
    StatRun.create([p0], ["mean"], collector.append)
    assert len(collector) == 1
    s: StatRun = collector[0]
    assert s.algorithm == algo0
    assert s.instance == inst0
    assert s.time_unit == tu
    assert s.f_name == fn
    assert s.n == 1
    assert s.stat_name == "mean"
    assert (s.stat == np.array([[10.0, 100.0],
                                [20.0, 90.0],
                                [30.0, 80.0],
                                [40.0, 70.0],
                                [50.0, 60.0],
                                [60.0, 50.0],
                                [70.0, 40.0]])).all()

    p1: Final[Progress] = Progress(algo0,
                                   inst0,
                                   0,
                                   np.array([5, 20, 35, 70, 90]),
                                   tu,
                                   np.array([67, 53, 45, 41, 33]),
                                   fn,
                                   None,
                                   True)
    collector.clear()
    StatRun.create([p1, p0], ["mean"], collector.append)
    assert len(collector) == 1
    s: StatRun = collector[0]
    assert s.algorithm == algo0
    assert s.instance == inst0
    assert s.time_unit == tu
    assert s.f_name == fn
    assert s.n == 2
    assert s.stat_name == "mean"
    assert (s.stat == np.array([[10.0, 0.5 * (100.0 + 67)],
                                [20.0, 0.5 * (53 + 90.0)],
                                [30.0, 0.5 * (80.0 + 53)],
                                [35, 0.5 * (80 + 45)],
                                [40.0, 0.5 * (70.0 + 45)],
                                [50.0, 0.5 * (60.0 + 45)],
                                [60.0, 0.5 * (50.0 + 45)],
                                [70.0, 0.5 * (40.0 + 41)],
                                [90, 0.5 * (40 + 33)]])).all()

    collector.clear()
    StatRun.create([p1, p0], ["max"], collector.append)
    assert len(collector) == 1
    s: StatRun = collector[0]
    assert s.algorithm == algo0
    assert s.instance == inst0
    assert s.time_unit == tu
    assert s.f_name == fn
    assert s.n == 2
    assert s.stat_name == "max"
    assert (s.stat == np.array([[10.0, 100.0],
                                [20.0, 90.0],
                                [30.0, 80.0],
                                [40.0, 70.0],
                                [50.0, 60.0],
                                [60.0, 50.0],
                                [70.0, 41.0],
                                [90, 40]])).all()
