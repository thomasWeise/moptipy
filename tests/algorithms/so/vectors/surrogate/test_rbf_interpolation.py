"""Test the sci-py surrogate assisted algorithm."""

from typing import Final

from numpy.random import default_rng
from scipy.special import comb

from moptipy.algorithms.so.vector.cmaes_lib import CMAES
from moptipy.algorithms.so.vector.surrogate.rbf_interpolation import (
    RBFInterpolation,
)
from moptipy.api.algorithm import Algorithm
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.on_vectors import (
    DIMENSIONS_FOR_TESTS,
    validate_algorithm_on_ackley,
)

#: the permitted RBF kernels
_RBF_KERNELS: Final[dict[str, str]] = {
    "linear": "l",
    "thin_plate_spline": "tps",
    "cubic": "c",
    "quintic": "q",
}


def test_rbf_cmaes_on_ackley() -> None:
    """Validate RBF-surrogate-based CMAES on Ackley's Function."""
    random = default_rng()

    def create(space: VectorSpace, _, __r=random) -> Algorithm:
        degree = int(__r.integers(2, 5))
        algo = CMAES(space)
        ks = list(_RBF_KERNELS.keys())
        kern = ks[__r.integers(len(ks))]
        fes_1 = int(__r.integers(1, 30))
        fes_2 = int(__r.integers(1, 100))
        rbfi = RBFInterpolation(
            space, algo, degree=degree, kernel=kern,
            fes_for_warmup=fes_1, fes_per_interpolation=fes_2)
        wu_fes = max(fes_1, int(comb(degree + space.dimension,
                                     space.dimension, exact=True)))
        assert str(rbfi) == \
               f"RBF{_RBF_KERNELS[kern]}{degree}_{wu_fes}_{fes_2}_{algo}"
        return rbfi

    validate_algorithm_on_ackley(
        create, uses_all_fes_if_goal_not_reached=True,
        dims=filter(lambda i: i > 1, DIMENSIONS_FOR_TESTS))
