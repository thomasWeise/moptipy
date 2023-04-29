"""
A unary operator flipping a pre-defined number of bits.

This is a unary operator with step size, i.e., an instance of
:class:`~moptipy.api.operators.Op1WithStepSize`. As such, it also receives
a parameter `step_size` when it is applied, which is from the closed range
`[0.0, 1.0]`. If `step_size=0`, then exactly 1 bit will be flipped. If
`step_size=1`, then all bits will be flipped. For all values of
`0<step_size<1`, we use the function
:func:`~moptipy.operators.tools.exponential_step_size` to extrapolate the
number of bits to flip.

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
from moptipy.operators.tools import exponential_step_size


class Op1FlipM(Op1WithStepSize):
    """A unary search operation that flips a specified number of `m` bits."""

    def op1(self, random: Generator, dest: np.ndarray, x: np.ndarray,
            step_size: float = 0.0) -> None:
        """
        Copy `x` into `dest` and flip exactly `m` bits.

        `step_size=0.0` will flip exactly `1` bit, `step_size=1.0` will flip
        all `n` bits. All other values are extrapolated by function
        :func:`~moptipy.operators.tools.exponential_step_size`. In between
        the two extremes, an exponential scaling
        (:func:`~moptipy.operators.tools.exponential_step_size`) is performed,
        meaning that for `n=10` bits, a step-size of `0.2` means flipping
        two bits, `0.4` means flipping three bits, `0.6` means flipping four
        bits, `0.7` means flipping five bits, `0.9` means flipping eight bits,
        `0.95` means flipping nine bits, and `1.0` means flipping all bits.
        In other words, a larger portion of the `step_size` range corresponds
        to making small changes while the large changes are all condensed at
        the higher end of the scale.

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
        >>> for ss in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        ...   op1.op1(rand, dst, src, ss)
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
            n, exponential_step_size(step_size, 1, n), False)
        dest[flips] ^= True  # flip the selected bits via xor

    def __str__(self) -> str:
        """
        Get the name of this unary operator.

        :return: "flipm"
        """
        return "flipm"
