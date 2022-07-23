"""A mock-up of an objective function."""

from math import inf, isfinite, nextafter
from typing import Union, Final, List, Iterable, Optional, Tuple, cast, Any, \
    Dict

import numpy as np
from numpy.random import Generator, default_rng

from moptipy.api.objective import Objective
from moptipy.mock.utils import make_ordered_list, sample_from_attractors
from moptipy.utils.logger import KeyValueLogSection, CSV_SEPARATOR
from moptipy.utils.nputils import is_np_int, is_np_float
from moptipy.utils.strings import num_to_str
from moptipy.utils.types import type_error


class MockObjective(Objective):
    """A mock-up of an objective function."""

    def __init__(self,
                 is_int: bool = True,
                 lb: Union[int, float] = -inf,
                 ub: Union[int, float] = inf,
                 fmin: Union[int, float, None] = None,
                 fattractors: Optional[Iterable[Union[
                     int, float, None]]] = None,
                 fmax: Union[int, float, None] = None,
                 seed: Optional[int] = None) -> None:
        """
        Create a mock objective function.

        :param is_int: is this objective function always integer?
        :param lb: the lower bound
        :param ub: the upper bound
        :param fmin: the minimum value this objective actually takes on
        :param fattractors: the attractor points
        :param fmax: the maximum value this objective actually takes on
        """
        if not isinstance(is_int, bool):
            raise type_error(is_int, "is_int", bool)
        #: is this objective integer?
        self.is_int: Final[bool] = is_int

        if seed is None:
            seed = int(default_rng().integers(0, 1 << 63))
        elif not isinstance(seed, int):
            raise type_error(seed, "seed", int)
        #: the random seed
        self.seed: Final[int] = seed

        #: the generator for setting up the mock objective
        random: Final[Generator] = default_rng(seed)
        #: the name of this objective function
        self.name: Final[str] = \
            f"mock{hex(random.integers(1, 100_000_000))[2:]}"

        if not isinstance(lb, (int, float)):
            raise type_error(lb, "lb", (int, float))
        if isfinite(lb) and is_int and not isinstance(lb, int):
            raise type_error(lb, f"finite lb @ is_int={is_int}", int)
        if not isinstance(ub, (int, float)):
            raise type_error(ub, "ub", (int, float))
        if isfinite(ub) and is_int and not isinstance(ub, int):
            raise type_error(ub, f"finite lb @ is_int={is_int}", int)
        if lb >= ub:
            raise ValueError(f"lb={lb} >= ub={ub} not permitted")
        #: the lower bound
        self.lb: Final[Union[int, float]] = lb
        #: the upper bound
        self.ub: Final[Union[int, float]] = ub

        if fmin is not None:
            if not isinstance(fmin, int if is_int else (int, float)):
                raise type_error(fmin, f"fmin[is_int={is_int}",
                                 int if is_int else (int, float))
            if fmin < lb:
                raise ValueError(f"fmin={fmin} < lb={lb}")
        if fmax is not None:
            if not isinstance(fmax, int if is_int else (int, float)):
                raise type_error(fmax, f"fmax[is_int={is_int}",
                                 int if is_int else (int, float))
            if fmax > ub:
                raise ValueError(f"fmax={fmax} < ub={ub}")
        if (fmin is not None) and (fmax is not None):
            if fmin >= fmax:
                raise ValueError(f"fmin={fmin} >= fmax={fmax}")

        values: List[Union[int, float, None]] = [lb, fmin]
        if fattractors is None:
            while True:
                values.append(None)
                if random.integers(2) <= 0:
                    break
        else:
            values.extend(fattractors)
        values.append(fmax)
        values.append(ub)

        values = make_ordered_list(values, is_int, random)
        if values is None:
            raise ValueError(
                f"could not create mock objective with lb={lb}, fmin={fmin}, "
                f"fattractors={fattractors}, fmax={fmax}, ub={ub}, "
                f"is_int={is_int}, and seed={seed}")

        #: the minimum value the function actually takes on
        self.fmin: Final[Union[int, float]] = values[1]
        #: the maximum value the function actually takes on
        self.fmax: Final[Union[int, float]] = values[-2]
        #: the mean value the function actually takes on
        self.fattractors: Final[Tuple[Union[int, float], ...]] =\
            cast(Tuple[Union[int, float], ...], tuple(values[2:-2]))
        #: the internal random number generator
        self.__random: Final[Generator] = random

    def sample(self) -> Union[int, float]:
        """
        Sample the mock objective function.

        :returns: the value of the mock objective function
        """
        return sample_from_attractors(self.__random, self.fattractors,
                                      self.is_int, self.lb, self.ub)

    def evaluate(self, x) -> Union[float, int]:
        """
        Return a mock objective value.

        :param x: the candidate solution
        :return: the objective value
        """
        seed: Optional[int] = None
        if hasattr(x, "__hash__") and (x.__hash__ is not None):
            seed = hash(x)
        elif isinstance(x, np.ndarray):
            seed = hash(x.tobytes())
        elif isinstance(x, List):
            seed = hash(str(x))

        if seed is None:
            random = self.__random
        else:
            random = default_rng(abs(seed))

        return sample_from_attractors(random, self.fattractors,
                                      self.is_int, self.lb, self.ub)

    def lower_bound(self) -> Union[float, int]:
        """
        Get the lower bound of the objective value.

        :return: the lower bound of the objective value
        """
        return self.lb

    def upper_bound(self) -> Union[float, int]:
        """
        Get the upper bound of the objective value.

        :return: the upper bound of the objective value
        """
        return self.ub

    def is_always_integer(self) -> bool:
        """
        Return `True` if :meth:`~evaluate` will always return an `int` value.

        :returns: `True` if :meth:`~evaluate` will always return an `int`
          or `False` if also a `float` may be returned.
        """
        return self.is_int

    def __str__(self):
        """Get the name of this mock objective function."""
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """Log the special parameters of tis mock objective function."""
        super().log_parameters_to(logger)
        logger.key_value("min", self.fmin)
        logger.key_value("attractors", CSV_SEPARATOR.join([
            num_to_str(n) for n in self.fattractors]))
        logger.key_value("max", self.fmax)
        logger.key_value("seed", self.seed)
        logger.key_value("is_int", self.is_int)

    @staticmethod
    def for_type(dtype: np.dtype) -> 'MockObjective':
        """
        Create a mock objective function with values bound by a given `dtype`.

        :param dtype: the numpy data type
        :returns: the mock objective function
        """
        if not isinstance(dtype, np.dtype):
            raise type_error(dtype, "dtype", np.dtype)

        random = default_rng()
        params: Dict[str, Any] = {}
        use_min = random.integers(2) <= 0
        use_max = random.integers(2) <= 0
        if not (use_min or use_max):
            if random.integers(5) <= 0:
                use_min = True
            else:
                use_max = True
        if is_np_int(dtype):
            params["is_int"] = True
            ii = np.iinfo(dtype)
            params["lb"] = lbi = max(int(ii.min), -(1 << 58))
            params["ub"] = ubi = min(int(ii.max), (1 << 58))
            if use_min:
                params["fmin"] = lbi + 1
            if use_max:
                params["fmax"] = ubi - 1

        if is_np_float(dtype):
            params["is_int"] = False
            fi = np.finfo(dtype)
            params["lb"] = lbf = max(float(fi.min), -1e300)
            params["ub"] = ubf = min(float(fi.max), 1e300)
            if use_min:
                params["fmin"] = nextafter(float(lbf + fi.eps), inf)
            if use_max:
                params["fmax"] = nextafter(float(ubf - fi.eps), -inf)

        if len(params) > 0:
            return MockObjective(**params)
        raise ValueError(f"unsupported dtype: {dtype}")
