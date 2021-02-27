from numpy.random import default_rng

from moptipy.api.operators import Op0
from moptipy.api.space import Space
from moptipy.tests.component import check_component


def check_op0(op0: Op0 = None,
              space: Space = None):
    """
    Check whether an object is a moptipy nullary operator.
    :param op0: the operator
    :param space: the space
    :raises ValueError: if `op0` is not a valid `Op0`
    """
    if not isinstance(op0, Op0):
        raise ValueError("Expected to receive an instance of Op0, but "
                         "got a '" + str(type(op0)) + "'.")
    check_component(component=op0)

    if not (space is None):
        random = default_rng()

        seen = set()
        max_count = 10

        x = space.create()
        for i in range(max_count):
            op0.op0(random, x)
            space.validate(x)
            seen.add(space.to_str(x))

        expected = max_count // 2
        if len(set) < expected:
            raise ValueError("It is expected that at least " + str(expected)
                             + " different elements will be created by "
                               "nullary search operator from "
                             + str(max_count) + " samples, but we only got "
                             + str(len(set)) + " different points.")
