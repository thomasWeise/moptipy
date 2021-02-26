from abc import abstractmethod
from moptipy.api.component import Component


class Encoding(Component):
    """
    A class to implement encodings, i.e., functions translating a
    search space to a solution space.
    """

    @abstractmethod
    def map(self, x, y):
        """
        Map a point `x` from the search space to a point `y`
        in the solution space.

        :param x: the point in the search space, remaining unchanged.
        :param y: the destination data structure for the point in the
            solution space, whose contents will be overwritten
        """
        raise NotImplementedError
