"""Test the end statistics parser."""

from typing import Final

from numpy.random import Generator, default_rng
from pycommons.io.temp import temp_file

from moptipy.evaluation.end_statistics import (
    EndStatistics,
    from_csv,
    from_end_results,
    to_csv,
)
from moptipy.mock.components import Experiment
from moptipy.mock.end_results import EndResults


def __test_write_read_end_stats(
        random: Generator,
        has_max_fes: bool, has_max_ms: bool,
        join_all_algorithms: bool, join_all_instances: bool,
        join_all_objectives: bool, join_all_encodings: bool) -> None:
    """Test writing and reading end results."""
    experiment: Final[Experiment] = Experiment.create(
        int(random.integers(1, 10)),
        int(random.integers(1, 10)),
        int(random.integers(1, 10)))
    max_fes: Final[int | None] = int(random.integers(
        1_000_000, 1_000_000_000)) if has_max_fes else None
    max_ms: Final[int | None] = int(random.integers(
        1_000_000, 1_000_000_000)) if has_max_ms else None
    end: Final[EndResults] = EndResults.create(experiment, max_fes, max_ms)
    stats: Final[list[EndStatistics]] = []

    from_end_results(end.results, stats.append,
                     join_all_algorithms, join_all_instances,
                     join_all_objectives, join_all_encodings)
    assert len(stats) > 0
    loaded: list[EndStatistics] = []

    with temp_file() as tf:
        to_csv(stats, tf)
        from_csv(tf, loaded.append)

    assert len(loaded) > 0
    stats.sort()
    loaded.sort()
    assert stats == loaded


def test_write_read_end_stats() -> None:
    """Test writing and reading end statistics."""
    random: Final[Generator] = default_rng()

    def __choices(ra: Generator) -> list[bool]:
        res = ra.integers(5)
        if res == 0:
            return [True, False]
        return [bool((res & 1) == 0)]

    for join_all_algorithms in __choices(random):
        for join_all_instances in __choices(random):
            for join_all_objectives in __choices(random):
                for join_all_encodings in __choices(random):
                    for has_max_fes in __choices(random):
                        for has_max_ms in __choices(random):
                            __test_write_read_end_stats(
                                random, has_max_fes, has_max_ms,
                                join_all_algorithms, join_all_instances,
                                join_all_objectives, join_all_encodings)
