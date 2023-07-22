"""
A meta-algorithm for model-assisted optimization using SciPy's Interpolation.

This algorithm wraps another numerical optimization algorithm `A` and proceeds
in two stages.
First, it samples and evaluates a set of initial points during the warmup
phase. These points are directly sampled by `A` on the original process, the
meta-algorithm just collects them.

Then, in the second stage, for each iteration, a model is constructed from
all previously sampled and evaluated points. The model is used for
interpolating the actual objective function.
In each step, the inner algorithm `A` is applied to this model. It strictly
works on the model and does not invoke the original objective. Instead, we
maintain the best point that `A` has sampled on the model based on the modeled
objective function. This best point is then evaluated on the actual objective
function. Together with its actual objective value, it is added to the set of
evaluated points. In the next step, a new model will be constructed based on
all the points we have now. This model is then the basis for the next
"simulated" run of `A`, and so on.

Thus, in the second stage, each execution of `A` on the model problem yields
one new point that is actually evaluated. The new point is used to create a
better model, and so on. If the models reflect the actual objective function
well, this may allow us to achieve better overall solution qualities or to
reduce the number of actual objective function evaluations to reach certain
goals.

However, this only works if a) we will not do too many actual objective
function evaluations (FEs) overall, as the memory requirement grows
quadratically with the number of FEs and b) if the dimensionality of the
problem is not too high, as the number of points needed to create a reasonably
accurate model rises with the dimensions of the search space.
"""

from math import inf
from typing import Callable, Final

import numpy as np
from scipy.interpolate import RBFInterpolator  # type: ignore
from scipy.special import comb  # type: ignore

from moptipy.algorithms.so.vector.surrogate._processes import (
    _SurrogateApply,
    _SurrogateWarmup,
)
from moptipy.api.algorithm import Algorithm, check_algorithm
from moptipy.api.process import Process
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT
from moptipy.utils.strings import num_to_str_for_name, sanitize_names
from moptipy.utils.types import check_int_range, type_error

#: the permitted RBF kernels
_RBF_KERNELS: Final[dict[str, str]] = {
    "linear": "l",
    "thin_plate_spline": "tps",
    "cubic": "c",
    "quintic": "q",
}


class RBFInterpolation(Algorithm):
    """
    A meta algorithm using an RBF-interpolation based surrogate model.

    This algorithm uses :class:`scipy.interpolate.RBFInterpolator` as
    interpolator surrogate model.
    """

    def __init__(self,
                 space: VectorSpace,
                 inner: Algorithm,
                 fes_for_warmup: int,
                 fes_per_interpolation: int,
                 kernel: str = "thin_plate_spline",
                 degree: int = 2,
                 name="RBF") -> None:
        """
        Create an interpolation-based surrogate algorithm.

        :param name: the base name of this algorithm
        :param inner: the algorithm to be applied in the inner optimization
            loop
        :param fes_for_warmup: the number of objective function evaluations to
            be spent on the initial warmup period
        :param fes_per_interpolation: the number of FEs to be performed
            for each interpolation run
        """
        super().__init__()

        if not isinstance(space, VectorSpace):
            raise type_error(space, "space", VectorSpace)
        if not isinstance(kernel, str):
            raise type_error(kernel, "kernel", str)
        if kernel not in _RBF_KERNELS:
            raise ValueError(
                f"kernel={kernel!r} not permitted, must be one "
                f"of {_RBF_KERNELS.keys()}.")

        degree = check_int_range(degree, "degree", -1, 20)
        dimensions = check_int_range(
            space.dimension, "space.dimensions", 1, 10_000)
        fes_for_warmup = check_int_range(
            fes_for_warmup, "fes_for_warmup", 1, 1_000_000_000)
        if degree >= 0:
            min_points: Final[int] \
                = int(comb(degree + dimensions, dimensions, exact=True))
            if min_points > fes_for_warmup:
                fes_for_warmup = min_points

        #: the vector space
        self._space = space
        #: the inner algorithm
        self.__inner = check_algorithm(inner)
        #: the warmup FEs
        self.__warmup_fes = fes_for_warmup
        #: the FEs per interpolation run
        self.__interpolation_fes = check_int_range(
            fes_per_interpolation, "fes_per_interpolation", 1,
            1_000_000_000_000)
        #: the name of this surrogate assisted
        self.__name = sanitize_names((
            f"{name}{_RBF_KERNELS[kernel]}{num_to_str_for_name(degree)}",
            str(fes_for_warmup), str(fes_per_interpolation), str(inner)))
        #: the kernel name
        self.__kernel: Final[str] = kernel
        #: the degree
        self.__degree: Final[int] = degree

    def __str__(self) -> str:
        """
        Get the name of this surrogate-assisted algorithm.

        :returns: the name of this surrogate assisted algorithm
        """
        return self.__name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this surrogate-assisted algorithm.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("warmupFEs", self.__warmup_fes)
        logger.key_value("interpolationFEs", self.__interpolation_fes)
        logger.key_value("kernel", self.__kernel)
        logger.key_value("degree", self.__degree)
        with logger.scope("inner") as inner:
            self.__inner.log_parameters_to(inner)
        with logger.scope("space") as space:
            self._space.log_parameters_to(space)

    def solve(self, process: Process) -> None:
        """
        Apply the surrogate-assisted optimization method to the given process.

        :param process: the process to solve
        """
        # fast calls
        should_terminate: Final[Callable[[], bool]] = process.should_terminate
        inner: Final[Callable[[Process], None]] = self.__inner.solve
        evaluate: Final[Callable[[np.ndarray], int | float]] = \
            process.evaluate
        init: Final[Callable[[], None]] = self.__inner.initialize
        uniform: Final[Callable[[
            np.ndarray, np.ndarray, int], np.ndarray]] = \
            process.get_random().uniform
        empty: Final[Callable[[int, np.dtype], np.ndarray]] \
            = np.empty

        # constants
        lb: Final[np.ndarray] = self._space.lower_bound
        ub: Final[np.ndarray] = self._space.upper_bound
        dim: Final[int] = self._space.dimension
        dtype: Final[np.dtype] = self._space.dtype
        run_fes: Final[int] = self.__interpolation_fes
        kernel: Final[str] = self.__kernel
        degree: Final[int] = self.__degree

# the containers for the points that we have sampled
        x: Final[list[np.ndarray]] = []  # the points that were sampled so far
        z: Final[list[int | float]] = []  # their objective values

# Perform the initial warm-up process. Here, the inner algorithm will directly
# be applied to the original problem. All the points that it samples are
# collected and will later be used to construct the model.
        with _SurrogateWarmup(process, self.__warmup_fes,
                              x.append, z.append) as p2:
            p2._fes_left = p2.max_fes  # type: ignore # store the budget
            p2._terminated = False  # type: ignore # not terminated yet
            init()  # initialize the inner algorithm
            inner(p2)  # apply the inner algorithm to the real model
        del p2

# Now, we have collected self.__warmup_fes points from the search space.

        if should_terminate():
            return

# We can now perform the optimization on the model. The model is constructed
# based on all points in the search space that were sampled and evaluated with
# the actual objective function. In each iteration, we apply the inner
# algorithm to the model from scratch. After it has terminated, then take the
# best point it found (based on the modeled objective function) and evaluate
# it with the actual objective function. This point and its objective value
# are then added to the internal list and used, together with all previous
# points, to construct the model for the next iteration.
        model: Final[_SurrogateApply] = _SurrogateApply(process, run_fes)

        while True:
            while True:
                # We always begin by building the surrogate model anew.
                # However, this may sometimes fail. Maybe a parameter matrix
                # becomes singular or whatever.
                try:
                    f: Callable[[np.ndarray], np.ndarray] = \
                        RBFInterpolator(np.array(x, dtype=DEFAULT_FLOAT),
                                        np.array(z, dtype=DEFAULT_FLOAT),
                                        kernel=kernel,
                                        degree=degree)
                    break  # success: quit innermost loop
                except:  # noqa # pylint: disable=[W0702]
                    # If we get here, the model construction has failed.
                    # This means that the points that we have collected are
                    # somehow insufficient.
                    # We try to fix this by sampling one additional point
                    # uniformly at random and evaluate it.
                    # If this does not exhaust the FEs that we have, we can
                    # then try again.
                    tmp = uniform(lb, ub, dim)
                    x.append(tmp)  # add random point to list of points
                    z.append(process.evaluate(tmp))  # and its objective value
                    if process.should_terminate():  # did we exhaust budget?
                        return  # yes ... so we return

            model._fes_left = run_fes  # assign the budget for the run
            model._terminated = False  # the run has not terminated
            model._evaluate = f  # forward evaluation to the model
            model._best_f = inf  # no best-so-far solution exists yet
            init()  # initialize the inner algorithm
            inner(model)  # apply the inner algorithm to the model
            tmp = empty(dim, dtype)  # allocate holder for result
            model.get_copy_of_best_x(tmp)  # get best solution
            z2 = evaluate(tmp)  # evaluate it on the actual problem
            if should_terminate():  # should we quit?
                return  # yes, so we return
            x.append(tmp)  # add the best solution to the list of points
            z.append(z2)  # and also add the objective value
            del f  # dispose old model
