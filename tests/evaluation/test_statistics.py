"""Test the execution of an experiment and parsing the log files the JSSP."""
import statistics

from moptipy.evaluation.statistics import Statistics


def __test_statistics(a: list[int | float],
                      geoprec=1e-12) -> None:
    stat = Statistics.create(a)
    assert stat is not None

    assert stat.minimum == min(a)
    assert stat.maximum == max(a)

    assert stat.median == statistics.median(a)
    assert abs(stat.mean_arith - statistics.mean(a)) <= 1e-14

    if len(a) <= 1:
        assert stat.stddev == 0
    else:
        assert abs(stat.stddev - statistics.stdev(a)) <= 1e-14

    if all(aa > 0 for aa in a):
        assert abs(stat.mean_geom - statistics.geometric_mean(a)) <= geoprec
    else:
        assert stat.mean_geom is None


def test_statistics() -> None:
    """Test the statistics."""
    __test_statistics([0, 0, 0, 0, 0, 0, 0])
    __test_statistics([0, 0, 0, 0, 0, 1, 0])
    __test_statistics([0, 0, 1, 0, 0, 0, 0])
    __test_statistics([1, 1, 0, 1, 1, 1, 1])
    __test_statistics([1, 1, 1, 1, 1, 1, 1])
    __test_statistics([0])
    __test_statistics([1])
    __test_statistics([0, 1])
    __test_statistics([1, 2])
    __test_statistics([3, 4])
    __test_statistics([1, 2, 3])
    __test_statistics([1.0, 2.0, 3.0])
    __test_statistics([0, 1, 2, 3])
    __test_statistics([1, 2, 3, 4])
    __test_statistics([4, 4, 4, 4, 4])
    __test_statistics([3.3, 4.5])
    __test_statistics([100000, 200000, 300000], 1e-10)
    __test_statistics([100000000, 200000000, 300000000,
                       4000000000], 1e-4)
    __test_statistics([100000000003, 200000000041, 300000000077,
                       40000000000001], 2e-3)
    __test_statistics([-5, 4, 2])
    __test_statistics([0, 4, 2])
    __test_statistics([5, 4, 2])

    assert Statistics.create([7, 8, 4, 2]) == \
           Statistics.create([2.0, 4.0, 8.0, 7.0])
