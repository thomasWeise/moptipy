"""
A unary operator flipping a pre-defined number of bits.

This is a unary operator with step size, i.e., an instance of
:class:`~moptipy.api.operators.Op1WithStepSize`. As such, it also receives
a parameter `step_size` when it is applied, which is from the closed range
`[0.0, 1.0]`. If `step_size=0`, then exactly 1 bit will be flipped. If
`step_size=1`, then all bits will be flipped. For all values of
`0<step_size<1`, we compute the number `m` of bits to be flipped as
`m = 1 + int(0.5 + step_size * (len(x) - 1))`, where `len(x)` is the length
of the bit strings.

Unary operators like this are often used in (1+1)-EAs or even in
state-of-the-art EAs such as the Self-Adjusting (1+(lambda,lambda)) GA.

1. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   doi: https://doi.org/10.1109/TEVC.2022.3191698,
   https://arxiv.org/pdf/2112.00229.pdf.
2. Eduardo Carvalho Pinto and Carola Doerr. *Towards a More Practice-Aware
   Runtime Analysis of Evolutionary Algorithms,* 2018,
   arXiv:1812.00493v1 [cs.NE] 3 Dec 2018. https://arxiv.org/abs/1812.00493.
"""
from typing import Final

import numpy as np
from numpy.random import Generator

from moptipy.api.operators import Op1WithStepSize


class Op1FlipM(Op1WithStepSize):
    """A unary search operation that flips a specified number of `m` bits."""

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray,
            step_size: float = 0.0) -> None:
        """
        Copy `x` into `dest` and flip exactly `m` bits.

        Here, `step_size = (bits-to-flip - 1) / (n - 1)`, meaning that
        `step_size=0.0` will flip exactly `1` bit and `step_size=1.0` will
        flip all `n` bits.

        :param self: the self pointer
        :param random: the random number generator
        :param dest: the destination array to receive the new point
        :param x: the existing point in the search space
        :param step_size: the number of bits to flip

        >>> op1 = Op1FlipM()
        >>> from numpy.random import default_rng as drg
        >>> rand = drg()
        >>> import numpy as npx
        >>> src = npx.zeros(10, bool)
        >>> dst = npx.zeros(10, bool)
        >>> for i in range(1, 11):
        ...   op1.op1(rand, dst, src, (i - 1.0) / (len(src) - 1))
        ...   print(sum(dst != src))
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        """
        np.copyto(dest, x)  # copy source to destination
        n: Final[int] = len(dest)  # get the number of bits
        flips: Final[np.ndarray] = random.choice(  # choose the bits
            n, 1 + int(0.5 + (step_size * (n - 1))), False)
        dest[flips] ^= True  # flip the selected bits via xor

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "flipm"
        """
        return "flipm"
