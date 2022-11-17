"""
The modified Greedy (2+1) EAmod Evolutionary Algorithm.

The Greedy (2+1) EAmod maintains a population of two individuals. Both
solutions, `x0` and `x1`, are intiially sampled independently and at random.
Each iteration consists of two steps, a crossover step and a mutation step.
The binary operator (crossover) is only applied if `x0` and `x1` have the
same objective value to produce offspring `z1`. If `x0` and `x1` have
different objective values, then `z1` is set to be the better of the two
parents. Then, the final offspring `z2` is derived by applying the unary
mutation operator. If `z2` is at least as good as the better one of `x0`
and `x1`, then it will be accepted. If `x1` and `x0` are as same as
good, one of them is randomly chosen to be replaced by `z2`. Otherwise,
the worse one is replaced.

This is the implementation of a general black-box version of the
Greedy (2+1) GAmod by Carvalho Pinto and Doerr. The original algorithm is a
genetic algorithm, i.e., an EA with a bit string-based search space and a
mutation operator flipping a Binomially distributed number of bits and
performing uniform crossover. Here we implement is a general EA where you can
plug in any crossover or mutation operator. Furthermore, the algorithm by
Carvalho Pinto and Doerr is, in turn, a modified version of Sudhold's
Greedy (2+1) GA, with some improvements for efficiency.

1. Eduardo Carvalho Pinto and Carola Doerr. Towards a More Practice-Aware
   Runtime Analysis of Evolutionary Algorithms. 2017. arXiv:1812.00493v1
   [cs.NE] 3 Dec 2018. [Online]. http://arxiv.org/pdf/1812.00493.pdf.
2. Dirk Sudholt. Crossover Speeds up Building-Block Assembly. In Proceedings
   of the 14th Annual Conference on Genetic and Evolutionary Computation
   (GECCO'12), July 7-11, 2012, Philadelphia, Pennsylvania, USA,
   pages 689-702. ACM, 2012. https://doi.org/10.1145/2330163.2330260.
"""
from typing import Callable, Final

from numpy.random import Generator

from moptipy.api.algorithm import Algorithm2
from moptipy.api.operators import Op0, Op1, Op2
from moptipy.api.process import Process


class GreedyTwoPlusOneEAmod(Algorithm2):
    """The modified Greedy (2+1) Evolutionary Algorithm."""

    def solve(self, process: Process) -> None:
        """
        Apply the EA to an optimization problem.

        :param process: the black-box process object
        """
        # Omitted for brevity: store function references in variables
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = self.op1.op1  # the unary operator
        op2: Final[Callable] = self.op2.op2  # the binary operator
        should_terminate: Final[Callable] = process.should_terminate
        equals: Final[Callable] = process.is_equal
        ri: Final[Callable] = random.integers

        x0 = create()  # allocate record for first solution
        op0(random, x0)  # randomly initialize first solution
        f0: int | float = evaluate(x0)  # evaluate first solution
        if should_terminate():  # should we quit?
            return  # yes.

        x1 = create()  # allocate record for first solution
        op0(random, x1)  # randomly initialize second solution
        f1: int | float = evaluate(x1)  # evaluate 2nd solution

        z1 = create()  # create record for result of binary operation
        z2 = create()  # create record for result of unary operation

        while not should_terminate():  # loop until budget is used up
            if f0 == f1:  # only perform binary operation if f0 == f1
                op2(random, z1, x0, x1)  # recombination
                p = z1  # input of unary operation comes from binary op
            else:
                if f0 > f1:  # swap x0 and x1 if x1 is better
                    f0, f1 = f1, f0  # swap objective values
                    x0, x1 = x1, x0  # swap solutions
                p = x0  # input of unary operation is best-so-far x
            op1(random, z2, p)  # apply unary operation
            if not (equals(z2, x0) or equals(z2, x1)):  # is z2 new?
                fnew = evaluate(z2)  # only then evaluate it
                if fnew <= f0:  # is it better or equal than x1
                    if (f1 > f0) or (ri(2) == 0):
                        z2, x1 = x1, z2  # swap z2 with x1 and
                        f1 = fnew  # and remember its quality
                    else:  # f1 == f2 and probability = 0.5
                        z2, x0 = x0, z2  # swap z2 with x0
                        f0 = fnew  # and remember its quality

    def __init__(self, op0: Op0, op1: Op1, op2: Op2) -> None:
        """
        Create the algorithm with nullary, unary, and binary search operator.

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param op2: the binary search operator
        """
        super().__init__("greedy2plus1EAmod", op0, op1, op2)
