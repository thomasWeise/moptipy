"""Test the General Configurable Memetic Algorithm."""

from numpy.random import Generator, default_rng

from moptipy.algorithms.modules.selections.fitness_proportionate_sus import (
    FitnessProportionateSUS,
)
from moptipy.algorithms.modules.selections.tournament_without_repl import (
    TournamentWithoutReplacement,
)
from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.algorithms.so.fitnesses.rank import Rank
from moptipy.algorithms.so.general_ma import GeneralMA
from moptipy.algorithms.so.rls import RLS
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.api.operators import Op0, Op1
from moptipy.api.process import Process
from moptipy.examples.jssp.instance import Instance
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.operators.op0_forward import Op0Forward
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.tests.on_bitstrings import validate_algorithm_on_onemax
from moptipy.tests.on_jssp import validate_algorithm_on_jssp
from moptipy.utils.types import type_error


class __MyRLS(RLS):
    """An internal RLS counting its invocations."""

    def __init__(self, op0: Op0, op1: Op1) -> None:
        super().__init__(op0, op1)
        #: the invocation counter for solve
        self.csolve = 0
        #: the invocation counter for init
        self.cinit = 0

    def solve(self, process: Process) -> None:
        """Solve the problem."""
        self.csolve += 1
        if self.cinit < 1:
            raise ValueError(
                f"initialize called {self.cinit} times but "
                f"solve called {self.csolve} times.")
        super().solve(process)

    def initialize(self) -> None:
        """Do the initialization."""
        self.cinit += 1
        super().initialize()


def __post(a: Algorithm, fes: int) -> None:
    if not isinstance(a, GeneralMA):
        raise type_error(a, "a", GeneralMA)
    r = a.ls
    if not isinstance(r, __MyRLS):
        raise type_error(r, "a.ls", __MyRLS)
    if r.cinit <= 0:
        raise ValueError("ls.initialize not called")
    r.cinit = 0
    invocations = fes / (a.ls_fes + 1)
    if invocations <= 0:
        return
    if invocations > r.csolve:
        raise ValueError(f"ls invoked {r.csolve} times but "
                         f"should be {invocations}!")
    r.csolve = 0


def test_general_ma_on_jssp_random() -> None:
    """Validate the general ma on the JSSP."""

    def create(instance: Instance, search_space: Permutations,
               objective: Objective) -> GeneralMA:
        assert isinstance(instance, Instance)
        assert isinstance(search_space, Permutations)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return GeneralMA(
            Op0Shuffle(search_space),
            Op2GeneralizedAlternatingPosition(search_space),
            __MyRLS(Op0Forward(), Op1Swap2()),
            int(random.integers(2, 12)), int(random.integers(1, 12)),
            int(random.integers(2, 32)),
            fitness=Direct() if random.integers(2) <= 0 else Rank(),
            survival=TournamentWithoutReplacement(2)
            if random.integers(2) <= 0 else FitnessProportionateSUS(),
            mating=TournamentWithoutReplacement(2)
            if random.integers(2) <= 0 else None)

    validate_algorithm_on_jssp(create, post=__post)


def test_general_ma_on_onemax_random() -> None:
    """Validate the general ma on the OneMax problem."""

    def create(bs: BitStrings, objective: Objective) -> GeneralMA:
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        random: Generator = default_rng()
        return GeneralMA(
            Op0Random(), Op2Uniform(),
            __MyRLS(Op0Forward(), Op1Flip1()),
            int(random.integers(2, 12)), int(random.integers(1, 12)),
            int(random.integers(2, 32)),
            fitness=Direct() if random.integers(2) <= 0 else Rank(),
            survival=TournamentWithoutReplacement(2)
            if random.integers(2) <= 0 else FitnessProportionateSUS(),
            mating=TournamentWithoutReplacement(2)
            if random.integers(2) <= 0 else None)

    validate_algorithm_on_onemax(create, post=__post)
