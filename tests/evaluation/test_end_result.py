"""Test the end results parser."""

from typing import Final

from numpy.random import Generator, default_rng
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.io.temp import temp_file

from moptipy.evaluation.end_results import EndResult, from_csv, to_csv


def __fake_str(random: Generator, has_already: bool) -> str | None:
    """
    Make a fake string.

    :param random: Generator instance.
    :param has_already: did we already have an end result?
    :returns: the fake string
    """
    if has_already and (random.integers(3) <= 0):
        return None
    s = ""
    while True:
        if str.__len__(s) > 0:
            s += "\n" if random.integers(2) <= 0 else CSV_SEPARATOR
        s += str(int(random.integers(1, 1_000_000)))
        if random.integers(3) <= 0:
            break
    return s


def __test_write_read_end_result(
        min_len: int,
        max_len: int,
        has_encoding: bool,
        has_goal_f: bool,
        has_max_fes: bool,
        has_max_ms: bool,
        has_x: bool = False,
        has_y: bool = False) -> None:
    """Test writing and reading end results."""
    random: Final[Generator] = default_rng()
    results: list[EndResult] = []

    fake_algos: Final[list[str]] = ["a1", "a2", "a3"]
    fake_insts: Final[list[str]] = ["i1", "i2", "i3"]
    fake_fs: Final[list[str]] = ["f1", "f2", "f3"]
    fake_encodings: Final[list[str | None]] = ["e1", "e2", "e3", "e4", None] \
        if has_encoding else [None]

    already_x: bool = False
    already_y: bool = False
    while True:
        length = len(results)
        if length >= max_len:
            break
        if (length > min_len) and (random.integers(10) <= 0):
            break

        best_f = int(random.integers(1_000)) \
            if random.integers(2) <= 0 else float(random.normal(0, 1))
        li_fe = int(random.integers(1, 1_000_000))
        li_ms = int(random.integers(1, 1_000_000))
        total_fe = int(random.integers(li_fe, li_fe + 1_000_000))
        total_ms = int(random.integers(li_ms, li_ms + 1_000_000))
        x: str | None = __fake_str(random, already_x) if has_x else None
        already_x = already_x or x is not None
        y: str | None = __fake_str(random, already_y) if has_y else None
        already_y = already_y or y is not None

        goal_f = None
        if has_goal_f:
            goal_f = int(random.integers(best_f - 1_000, best_f - 1)) \
                if isinstance(best_f, int) else (
                float(random.uniform(best_f - 1e3, best_f - 1.0)))

        max_fes = None
        if has_max_fes:
            max_fes = int(random.integers(total_fe, total_fe + 1_000_000))

        max_ms = None
        if has_max_ms:
            max_ms = int(random.integers(total_ms, total_ms + 1_000_000))

        results.append(
            EndResult(random.choice(fake_algos),
                      random.choice(fake_insts),
                      random.choice(fake_fs),
                      random.choice(fake_encodings),
                      int(random.integers(0, 1_000_000_000_000)),
                      best_f, li_fe, li_ms, total_fe, total_ms, goal_f,
                      max_fes, max_ms, x, y))

    assert len(results) > 0
    loaded: list[EndResult] = []
    with temp_file() as tf:
        to_csv(results, tf)
        loaded.extend(from_csv(tf))

    assert len(loaded) > 0
    results.sort()
    loaded.sort()
    assert results == loaded
    for i, a in enumerate(results):
        b = loaded[i]
        assert (a.x is None and b.x is None) or (a.x == b.x)
        assert (a.y is None and b.y is None) or (a.y == b.y)


def test_write_read_end_result() -> None:
    """Test writing and reading end results."""
    choices: Final[tuple[bool, bool]] = True, False
    for min_len in (1, 2, 3, 10):
        for max_len in (min_len, min_len + 1, min_len + 100000):
            for has_encoding in choices:
                for has_goal_f in choices:
                    for has_max_fes in choices:
                        for has_max_ms in choices:
                            for has_x in choices:
                                for has_y in choices:
                                    __test_write_read_end_result(
                                        min_len, max_len,
                                        has_encoding, has_goal_f,
                                        has_max_fes, has_max_ms,
                                        has_x, has_y)
