"""
A simple implementation of a Plant Propagation Algorithm (PPA).

This is a simple implementation of the Plant Propagation Algorithm, PPA for
short, with some tweaks and modifications.
Our PPA implementation works as follows:

1. It starts with a set of :attr:`~moptipy.algorithms.so.ppa.PPA.m` randomly
   sampled solutions in the list `lst`. Each solution `x` is evaluated and its
   objective value `f(x)` is remembered.
2. In the main loop...

   a. First, the range `[fmin,fmax]` of the objective values of the first
      :attr:`~moptipy.algorithms.so.ppa.PPA.m` solutions in `lst` is
      determined. We set `frange = fmax - fmin`, where `fmax` is the largest
      objective value of any of the first `m` solutions in `lst` and `fmin`
      is the smallest one. If `frange > 0`, then the fitness `z(x)` of each
      element be `(f(x) - fmin) / frange`. Otherwise, i.e., if all solutions
      in `lst` have the same objective value, we set `z(x)` to be a random
      number uniformly distributed in `[0,1)` and drawn separately for each
      solution.
   b. For each of the first :attr:`~moptipy.algorithms.so.ppa.PPA.m` solutions
      `x` in `lst`, we create `1 + int(nmax * r * (1 - z(x)))` offspring,
      where :attr:`~moptipy.algorithms.so.ppa.PPA.nmax` is the maximum number
      of offspring per solution and `r` be again an independently drawn random
      number uniformly distributed in `[0,1)`. In other words, solutions with
      a fitness close to zero will produce more offspring. If the solutions in
      the list `lst` have different objective values, then this means that
      better solutions produce more offsprings.
      Each such offspring is the result of the application of a unary operator
      with step size, i.e., an instance of
      :class:`~moptipy.api.operators.Op1WithStepSize`. The step size is set to
      `r * max_step * z(x)`, where `r` again is a freshly and independently
      drawn random number uniformly distributed in `[0,1)`. This means that
      better solutions are modified with smaller step sizes and worse
      solutions are modified more strongly.
      :attr:`~moptipy.algorithms.so.ppa.PPA.max_step` is a parameter of the
      algorithm that determines the maximum permissible step size. It is
      always from the interval `[0,1]`.
      Examples for such operators are given in
      :mod:`~moptipy.operators.permutations.op1_swap_exactly_n`,
      :mod:`~moptipy.operators.permutations.op1_swap_try_n`, or
      :mod:`~moptipy.operators.bitstrings.op1_flip_m`.
      The new solutions are appended into `lst` and their objective values are
      computed.
   c. The list is then sorted by objective values in ascending order, meaning
      that the best solutions are up front.

The main differences between this procedure and the "standard-PPA" are as
follows:

A. The algorithm is implemented for minimization and all equations are
   modified accordingly.
B. After normalizing the objective values in the population, the `tanh`-based
   scaling is *not* applied.
   Instead, the normalized objective values, where `0` is best and `1` is
   worst, are used directly to determine the number of offspring per record
   and the step length.
C. The fitness of a record equals its normalized objective value
   (in `[0, 1]`), unless all records have the same objective value, in which
   case the fitness of each record is set to a random number uniformly
   distributed in `[0, 1)`.
   If all elements in the population have the same objective value,
   normalizing is not possible as it would lead to a division by zero.
   One could use a constant value, say `0.5`, in this case, but there is no
   guarantee that this would be a good choice.
   We therefore use random values from `[0, 1)` instead.
   These may sometimes be suitable, sometimes not.
   But at least they likely are not *always* a bad choice, which might happen
   in some scenarios with `0.5` or any other constant.
D. The decisions regarding the number of offspring per selected record and the
   step-width of the search moves are made only based on this fitness (and,
   again, not on the `tanh` scaling which is not used).
   Since we normally do not know the characteristics of the objective function
   in advance, I think that we also often do not know whether a `tanh` scaling
   (that emphasizes objective values close to the best and close to the worst)
   is necessary or a good idea.
   It could be good in some cases, but it might as well be a bad choice in
   others.
   For now, I have thus not implemented this and just use the raw normalized
   objective values.
E. As unary operators, we employ instances of the class
   :class:`~moptipy.api.operators.Op1WithStepSize`, which provides a unary
   operator with a step size between `0` (smallest possible modification) to
   `1` (largest possible modification) and will scale appropriately between
   the two extremes.
   Often, instances of this class will determine the number or magnitude of
   changes based on an exponential scaling (see
   :func:`~moptipy.operators.tools.exponential_step_size`) of the step length.
   The idea is that small step sizes should be emphasized and that really big
   step sizes are often rarely needed.
   This thus effectively takes the place of the `tanh` scaling.
F. Maximum step lengths, i.e., the parameter
   :attr:`~moptipy.algorithms.so.ppa.PPA.max_step`, are not always explicitly
   used in some of the papers.

In order to understand the behavior of the algorithm, consider the following
case. Assume that we set the maximum number
(:attr:`~moptipy.algorithms.so.ppa.PPA.nmax`) of offspring per solution to `1`
and the number :attr:`~moptipy.algorithms.so.ppa.PPA.m` of solutions to
survive selection to `1` as well. In this case, the PPA has exactly the same
"population structure" as the Randomized Local Search
(:mod:`~moptipy.algorithms.so.rls`), namely it preserves the best-so-far
solution and generates one new solution in each step, which then competes with
that best-so-far solution. The two algorithms then only differ in their search
operator: The step-length of the unary operator used in PPA depends on the
relative objective value and the one of the RLS does not. However, there are
many situations where the two could still be equivalent. For example, if the
current and new solution have different objective values, normalizing the
objective value will mean that the best of the two has normalized objective
value "0". This equates to the shortest possible step length. In this case,
for example, the step-length based operator
:mod:`~moptipy.operators.permutations.op1_swap_try_n` behaves exactly like the
:mod:`~moptipy.operators.permutations.op1_swap2` operator and the step-length
based :mod:`~moptipy.operators.bitstrings.op1_flip_m` operator behaves like
the :mod:`~moptipy.operators.bitstrings.op1_flip1`.
Of course, if both the current and the new solution have the same objective
value, then we use a random number from `[0,1)` as normalized objective value,
so the operators would not behave the same. Then again, one could set the
maximum step length :attr:`~moptipy.algorithms.so.ppa.PPA.max_step` to `0`.
In this case, the step length is always zero and most of our step-length based
operations will behave like fixed small step-length based counterparts, as
mentioned above. So in other words, if we set both
:attr:`~moptipy.algorithms.so.ppa.PPA.m` and
:attr:`~moptipy.algorithms.so.ppa.PPA.nmax` to `1` and set
:attr:`~moptipy.algorithms.so.ppa.PPA.max_step` to `0`, our PPA behaves like
:mod:`~moptipy.algorithms.so.rls` (if the search operators are appropriately
chosen).

Now :mod:`~moptipy.algorithms.so.rls` is also known as the (1+1) EA
and indeed, it is a special case of the (mu+lambda) EA implemented in
:mod:`~moptipy.algorithms.so.ea`.
I think with some appropriate settings of the parameter, we can probably
construct some setups of both algorithms with larger populations that should
be equivalent or close-to-equivalent in the big picture.

Below, you can find references on the PPA.

1. Abdellah Salhi and Eric Serafin Fraga. Nature-Inspired Optimisation
   Approaches and the New Plant Propagation Algorithm. *Proceeding of the
   International Conference on Numerical Analysis and Optimization
   (ICeMATH'2011),* June 6-8, 2011, Yogyakarta, Indonesia, volume 1,
   pages K2-1--K2-8. ISBN: 978-602-98919-1-1.
   https://doi.org/10.13140/2.1.3262.0806.
   https://repository.essex.ac.uk/9974/1/paper.pdf.
2. Misha Paauw and Daan van den Berg. Paintings, Polygons and Plant
   Propagation. In Anikó Ekárt, Antonios Liapis, and María Luz Castro Pena,
   editors, *Proceedings of the 8th International Conference on Computational
   Intelligence in Music, Sound, Art and Design (EvoMUSART'19, Part of
   EvoStar)*, April 24-26, 2019, Leipzig, Germany, Lecture Notes in Computer
   Science (LNCS), volume 11453, pages 84-97. ISBN: 978-3-030-16666-3. Cham,
   Switzerland: Springer. https://doi.org/10.1007/978-3-030-16667-0_6.
   https://www.researchgate.net/publication/332328080.
3. Muhammad Sulaiman, Abdellah Salhi, Eric Serafin Fraga, Wali Khan Mashwa,
   and Muhammad M. Rashi. A Novel Plant Propagation Algorithm: Modifications
   and Implementation. *Science International (Lahore)* 28(1):201-209, #2330,
   January/February 2016. http://www.sci-int.com/pdf/4066579081%20a%20201-\
209%20PPA%20Science%20international_Wali.pdf.
   https://arxiv.org/pdf/1412.4290.pdf
4. Hussein Fouad Almazini, Salah Mortada, Hassan Fouad Abbas Al-Mazini, Hayder
   Naser Khraibet AL-Behadili, and Jawad Alkenani. Improved Discrete Plant
   Propagation Algorithm for Solving the Traveling Salesman Problem. *IAES
   International Journal of Artificial Intelligence (IJ-AI)* 11(1):13-22.
   March 2022. http://doi.org/10.11591/ijai.v11.i1.pp13-22.
   https://www.researchgate.net/publication/357484222.
5. Birsen İrem Selamoğlu and Abdellah Salhi. The Plant Propagation Algorithm
   for Discrete Optimisation: The Case of the Travelling Salesman Problem.
   In Xin-She Yang, editor, *Nature-Inspired Computation in Engineering,*
   pages 43-61. Studies in Computational Intelligence (SCI), Volume 637.
   March 2016. Cham, Switzerland: Springer.
   https://doi.org/10.1007/978-3-319-30235-5_3.
   https://www.researchgate.net/publication/299286896.
6. Marleen de Jonge and Daan van den Berg. Parameter Sensitivity Patterns in
   the Plant Propagation Algorithm. In Juan Julián Merelo Guervós,
   Jonathan M. Garibaldi, Christian Wagner, Thomas Bäck, Kurosh Madani, and
   Kevin Warwick, editors, *Proceedings of the 12th International Joint
   Conference on Computational Intelligence* (IJCCI'20), November 2-4, 2020,
   Budapest, Hungary, pages 92-99. Setúbal, Portugal: SciTePress.
   https://doi.org/10.5220/0010134300920099.
   https://www.researchgate.net/publication/346829569.
7. Ege de Bruin. Escaping Local Optima by Preferring Rarity with the
   Frequency Fitness Assignment. Master's Thesis at Vrije Universiteit
   Amsterdam, Amsterdam, the Netherlands. 2022.
8. Wouter Vrielink and Daan van den Berg. Parameter control for the Plant
   Propagation Algorithm. In Antonio M. Mora and Anna Isabel Esparcia-Alcázar,
   editors, *Late-Breaking Abstracts of EvoStar'21*, April 7-9, 2021, online
   conference. https://arxiv.org/pdf/2106.11804.pdf.
   https://www.researchgate.net/publication/350328314.
9. Levi Koppenhol, Nielis Brouwer, Danny Dijkzeul, Iris Pijning, Joeri
   Sleegers, and Daan van den Berg. Exactly Characterizable Parameter Settings
   in a Crossoverless Evolutionary Algorithm. In Jonathan E. Fieldsend and
   Markus Wagner, editors, Genetic and Evolutionary Computation Conference
   (GECCO'22) Companion Volume, July 9-13, 2022, Boston, MA, USA,
   pages 1640-1649. New York, NY, USA: ACM.
   https://doi.org/10.1145/3520304.3533968.
   https://www.researchgate.net/publication/362120506.
"""
from math import isfinite
from typing import Callable, Final, cast

from numpy.random import Generator

from moptipy.algorithms.so.record import Record
from moptipy.api.algorithm import Algorithm1
from moptipy.api.operators import Op0, Op1WithStepSize
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import PART_SEPARATOR, num_to_str_for_name
from moptipy.utils.types import check_int_range, type_error


# start book
class PPA(Algorithm1):
    """The Plant Propagation Algorithm (PPA)."""

    def solve(self, process: Process) -> None:
        """
        Apply the PPA to an optimization problem.

        :param process: the black-box process object
        """
        m: Final[int] = self.m  # m: the number of best solutions kept
        nmax: Final[int] = self.nmax  # maximum offspring per solution
        list_len: Final[int] = (nmax + 1) * m
        # initialization of some variables omitted in book for brevity
        # end book
        random: Final[Generator] = process.get_random()  # random gen
        create: Final[Callable] = process.create  # create x container
        evaluate: Final[Callable] = process.evaluate  # the objective
        op0: Final[Callable] = self.op0.op0  # the nullary operator
        op1: Final[Callable] = cast(Op1WithStepSize,
                                    self.op1).op1  # the unary operator
        should_terminate: Final[Callable] = process.should_terminate
        r01: Final[Callable[[], float]] = cast(  # random floats
            Callable[[], float], random.random)
        max_step: Final[float] = self.max_step
        # start book
        # create list of m random records and enough empty records
        lst: Final[list] = [None] * list_len  # pre-allocate list
        f: int | float = 0  # variable to hold objective values
        for i in range(list_len):  # fill list of size m*nmax
            x = create()  # by creating point in search space
            if i < m:  # only the first m records are initialized by
                op0(random, x)  # applying nullary operator = randomize
                if should_terminate():  # should we quit?
                    return   # computational budget exhausted -> quit
                f = evaluate(x)  # continue? ok, evaluate new solution
            lst[i] = Record(x, f)  # create and store record

        it: int = 0  # the iteration counter
        while True:  # lst: keep 0..mu-1, overwrite mu..mu+lambda-1
            it = it + 1  # step iteration counter
            fmin = fmax = lst[0].f  # get range of objective values
            for i in range(m):  # iterate over selected individuals
                fval = lst[i].f  # get objective value
                if fval < fmin:  # is it less than minimum?
                    fmin = fval  # yes -> update the minimum
                elif fval > fmax:  # no! is it more than maximum then?
                    fmax = fval  # yes -> update maximum
            frange = fmax - fmin  # compute the range of objective
            all_same = (not isfinite(frange)) or (frange <= 0.0)
            total = m  # the total population length (so far: m)
            for i in range(m):  # generate offspring for each survivor
                rec = lst[i]  # get parent record
                fit = r01() if all_same else ((rec.f - fmin) / frange)
                x = rec.x  # the parent x
                for _ in range(1 + int((1.0 - fit) * r01() * nmax)):
                    if should_terminate():  # should we quit?
                        return  # yes - then return
                    dest = lst[total]  # get next destination record
                    total = total + 1  # remember we have now one more
                    dest.it = it  # set iteration counter
                    op1(random, dest.x, x, fit * max_step * r01())
                    dest.f = evaluate(dest.x)  # evaluate new point
            ls = lst[0:total]  # get sub-list of elements in population
            ls.sort()  # sort these used elements
            lst[0:total] = ls  # write the sorted sub-list back
# end book

    def __init__(self, op0: Op0, op1: Op1WithStepSize, m: int = 30,
                 nmax: int = 5, max_step: float = 0.3,
                 name: str = "ppa") -> None:
        """
        Create the Plant Propagation Algorithm (PPA).

        :param op0: the nullary search operator
        :param op1: the unary search operator
        :param m: the number of best solutions to survive in each generation
        :param nmax: the maximum number of offspring per solution
        :param name: the base name of the algorithm
        """
        if not isinstance(op1, Op1WithStepSize):
            raise type_error(op1, "op1", Op1WithStepSize)

        #: the number of records to survive in each generation
        self.m: Final[int] = check_int_range(m, "m", 1, 1_000_000)
        #: the maximum number of offsprings per solution per iteration
        self.nmax: Final[int] = check_int_range(
            nmax, "nmax", 1, 1_000_000)
        if not isinstance(max_step, float):
            raise type_error(max_step, "max_step", float)
        if (not isfinite(max_step)) or (max_step < 0.0) or (max_step > 1.0):
            raise ValueError(f"max_step={max_step}, but must be in [0,1].")
        #: the maximum step length
        self.max_step: Final[float] = max_step

        name = f"{name}{PART_SEPARATOR}{m}{PART_SEPARATOR}{nmax}"
        if max_step != 1.0:
            name = f"{name}{PART_SEPARATOR}{num_to_str_for_name(max_step)}"
        super().__init__(name, op0, op1)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to a logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("m", self.m)
        logger.key_value("nmax", self.nmax)
        logger.key_value("maxStep", self.max_step)
