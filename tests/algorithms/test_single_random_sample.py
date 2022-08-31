"""Test the single random sample algorithm."""
from moptipy.algorithms.single_random_sample import SingleRandomSample
from moptipy.api.objective import Objective
from moptipy.examples.bitstrings.onemax import OneMax
from moptipy.operators.bitstrings.op0_random import Op0Random
from moptipy.spaces.bitstrings import BitStrings
from moptipy.tests.on_bitstrings import validate_algorithm_on_bitstrings, \
    dimensions_for_tests


def test_single_random_sample_on_onemax():
    """Validate the single random sample algorithm on OneMax."""

    def create(bs: BitStrings, objective: Objective):
        assert isinstance(bs, BitStrings)
        assert isinstance(objective, Objective)
        return SingleRandomSample(Op0Random())

    for i in dimensions_for_tests():
        validate_algorithm_on_bitstrings(
            objective=OneMax,
            algorithm=create,
            dimension=i,
            max_fes=100000000,
            required_result=i)
