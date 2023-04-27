"""Test the direct fitness assignment strategy."""

from numpy.random import default_rng

from moptipy.algorithms.so.fitness import FRecord
from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.tests.on_bitstrings import validate_fitness_on_bitstrings


def test_direct_on_bit_strings() -> None:
    """Test the direct assignment process on bit strings."""
    validate_fitness_on_bitstrings(fitness=Direct())


def test_direct_on_random() -> None:
    """Test the direct assignment process on random values."""
    rand = default_rng()
    fitness = Direct()
    for _ in range(10):
        for i in range(rand.integers(1, 10)):
            lst = []
            mix_mode = rand.integers(4)
            def_val = rand.integers(-100, 100) \
                if rand.integers(2) <= 0 else rand.normal()
            if mix_mode <= 0:
                def __f(_r=rand, _=def_val) -> int | float:
                    return _r.integers(-100, 100)
            elif mix_mode <= 1:
                def __f(_r=rand, _=def_val) -> int | float:
                    return _r.normal()
            elif mix_mode <= 2:
                def __f(_r=rand, _=def_val) -> int | float:
                    if _r.integers(2) <= 0:
                        return _r.integers(-100, 100)
                    return _r.normal()
            else:
                def __f(_=rand, v=def_val) -> int | float:
                    return v

            for _j in range(i + 1):
                r = FRecord(None, __f())
                r.it = rand.integers(1, 1000)
                lst.append(r)

            fitness.assign_fitness(lst, rand)
            for p in lst:
                assert p.fitness is p.f
