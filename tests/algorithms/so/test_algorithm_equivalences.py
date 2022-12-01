"""Test several algorithm equivalences."""

from typing import Callable, Final, cast

from numpy.random import Generator, default_rng

from moptipy.algorithms.modules.selections.tournament_with_repl import (
    TournamentWithReplacement,
)
from moptipy.algorithms.modules.selections.tournament_without_repl import (
    TournamentWithoutReplacement,
)
from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.ea import EA
from moptipy.algorithms.so.fea1plus1 import FEA1plus1
from moptipy.algorithms.so.fitness import FRecord
from moptipy.algorithms.so.fitnesses.direct import Direct
from moptipy.algorithms.so.fitnesses.ffa import FFA
from moptipy.algorithms.so.fitnesses.rank import Rank
from moptipy.algorithms.so.general_ea import GeneralEA
from moptipy.algorithms.so.record import Record
from moptipy.algorithms.so.rls import RLS
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.operators.bitstrings.op1_flip1 import Op1Flip1
from moptipy.operators.bitstrings.op2_uniform import Op2Uniform
from moptipy.tests.on_bitstrings import verify_algorithms_equivalent


def test_opoea_equals_rls() -> None:
    """Test whether the (1+1)-EA performs exactly as RLS."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()

    verify_algorithms_equivalent([
        lambda bs, f: RLS(op0, op1),
        lambda bs, f: EA(op0, op1, op2, 1, 1, 0.0),
        lambda bs, f: GeneralEA(op0, op1, op2, 1, 1, 0.0),
    ])


def test_ea_equals_random_sampling_if_mu_gte_max_fes() -> None:
    """Test whether the EA equals random sampling if max_fes<=mu."""
    op0: Final[Op0] = Op0Random()
    verify_algorithms_equivalent([
        lambda bs, f: EA(op0, Op1Flip1(), Op2Uniform(), 100, 100, 0.0),
        lambda bs, f: RandomSampling(op0),
    ], max_fes=100)


def _int_0(_: int) -> int:
    """
    Return an integer with value `0`.

    :retval 0: always
    """
    return 0


def _float_0() -> float:
    """
    Return a float with value `0.0`.

    :retval 0.0: always
    """
    return 0.0


class __EAC(EA):
    """A slightly modified version of the EA."""

    def solve(self, process: Process) -> None:
        """
        Apply the EA to an optimization problem.

        :param process: the black-box process object
        """
        mu: Final[int] = self.mu  # mu: number of best solutions kept
        lst_size: Final[int] = mu + self.lambda_  # size = mu + lambda
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        br: Final[float] = self.br  # the rate at which to use op2
        should_terminate: Final[Callable] = process.should_terminate
        r0i: Final[Callable[[int], int]] = cast(  # only if m > 1, we
            Callable[[int], int], random.integers  # need random
            if mu > 1 else _int_0)  # indices
        r01: Final[Callable[[], float]] = cast(  # only if 0<br<1, we
            Callable[[], float],  # need random floats
            random.random if 0 < br < 1 else _float_0)
        # create list of mu random records and lambda empty records
        lst: Final[list] = [None] * lst_size  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(lst_size):  # fill list of size mu+lambda
            x = create()  # by creating point in search space
            if i < mu:  # only the first mu records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return  # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        # fix sorting
        lst[0:mu] = sorted(lst[0:mu], key=lambda r: (r.f, -r.it))

        it: int = 0
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it += 1  # step iteration counter
            for oi in range(mu, lst_size):  # for all lambda offspring
                if should_terminate():  # only continue if we still...
                    return  # have sufficient budget ... otherwise quit
                dest: Record = lst[oi]  # pick destination record
                x = dest.x  # the destination "x" value
                dest.it = it  # remember iteration of solution creation

                # This is the difference to the normal EA:
                # We _FIRST_ decide whether to do the binary operation and
                # _AFTERWARDS_ pick the first random parent...
                do_binary: bool = r01() < br
                sx = lst[r0i(mu)].x  # pick a random source record
                if do_binary:  # apply binary operator at rate br
                    sx2 = sx  # second source "x"
                    while sx2 is sx:  # must be different from sx
                        sx2 = lst[r0i(mu)].x  # get second record
                    op2(random, x, sx, sx2)  # apply binary op
                    dest.f = evaluate(x)  # evaluate new point
                    continue  # below is "else" part with unary operat.
                op1(random, x, sx)  # apply unary operator
                dest.f = evaluate(x)  # evaluate new point
            lst.sort()  # best records come first, ties broken by age


def test_general_ea_equals_ea() -> None:
    """Ensure that the EAs with and without fitness are identical."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()
    random: Final[Generator] = default_rng()
    for _ in range(3):
        mu: int = int(random.integers(2, 10))
        lambda_: int = int(random.integers(1, 10))
        br: float = float(random.uniform(0.1, 0.9))

        verify_algorithms_equivalent([
            lambda bs, f, mx=mu, lx=lambda_, bx=br: __EAC(
                op0, op1, op2, mx, lx, bx),
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx),
        ])


def test_general_ea_with_ffa_equals_fea() -> None:
    """Ensure that the FEA and the EA with FFA are identical."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()

    verify_algorithms_equivalent([
        lambda bs, f: FEA1plus1(op0, op1),
        lambda bs, f: GeneralEA(op0, op1, fitness=FFA(f)),
    ])


class __SDirect(Direct):
    """The sorting direct."""

    def assign_fitness(self, p: list[FRecord], random: Generator) -> None:
        for rec in p:  # first copy rec.f to rec.fitness
            rec.fitness = rec.f  # because then we can easily sort
        p.sort()  # sort based on objective values
        super().assign_fitness(p, random)


def test_general_ea_under_order_preserving_fitnesses() -> None:
    """Test that GeneralEA selects the same under order-preserving fitness."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()
    random: Final[Generator] = default_rng()
    for _ in range(3):
        mu: int = int(random.integers(2, 10))
        lambda_: int = int(random.integers(1, 10))
        br: float = float(random.uniform(0.1, 0.9))

        verify_algorithms_equivalent([
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=__SDirect()),
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=Rank()),
        ])


def test_general_ea_under_order_preserving_fitnesses_and_tournament() -> None:
    """Test that tournaments select same under order-preserving fitness."""
    op0: Final[Op0] = Op0Random()
    op1: Final[Op1] = Op1Flip1()
    op2: Final[Op2] = Op2Uniform()
    random: Final[Generator] = default_rng()
    for _ in range(3):
        mu: int = int(random.integers(2, 10))
        lambda_: int = int(random.integers(1, 10))
        br: float = float(random.uniform(0.1, 0.9))

        verify_algorithms_equivalent([
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=__SDirect(),
                survival=TournamentWithReplacement()),
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=Rank(),
                survival=TournamentWithReplacement()),
        ])
    for _ in range(3):
        mu: int = int(random.integers(2, 10))
        lambda_: int = int(random.integers(1, 10))
        br: float = float(random.uniform(0.1, 0.9))

        verify_algorithms_equivalent([
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=__SDirect(),
                survival=TournamentWithoutReplacement()),
            lambda bs, f, mx=mu, lx=lambda_, bx=br: GeneralEA(
                op0, op1, op2, mx, lx, bx, fitness=Rank(),
                survival=TournamentWithoutReplacement()),
        ])
