"""Convert moptipy data to IOHanalyzer data."""

import sys
from typing import Final, Optional, Dict, Union, Callable, List, Tuple

import numpy as np

from moptipy.evaluation.base import check_f_name, TIME_UNIT_FES, F_NAME_RAW
from moptipy.evaluation.progress import Progress
from moptipy.utils.console import logger
from moptipy.utils.help import help_screen
from moptipy.utils.path import Path
from moptipy.utils.strings import num_to_str
from moptipy.utils.types import type_error


def __prefix(s: str) -> str:
    """
    Return `xxx` if `s` is of the form `xxx_i` and `i` is `int`.

    :param s: the function name
    :return: the dimension
    """
    idx = s.rfind("_")
    if idx > 0:
        try:
            i = int(s[idx + 1:])
            if i > 0:
                return s[:idx].strip()
        except ValueError:
            pass  # ignore error
    return s


def __int_suffix(s: str) -> int:
    """
    Return `i` if `s` is of the form `xxx_i` and `i` is `int`.

    This function tries to check if the name

    :param s: the function name
    :return: the dimension
    """
    idx = s.rfind("_")
    if idx > 0:
        try:
            i = int(s[idx + 1:])
            if i > 0:
                return i
        except ValueError:
            pass  # ignore error
    return 1


def moptipy_to_ioh_analyzer(
        results_dir: str,
        dest_dir: str,
        inst_name_to_func_id: Callable[[str], str] = __prefix,
        inst_name_to_dimension: Callable[[str], int] = __int_suffix,
        inst_name_to_inst_id: Callable[[str], int] = lambda x: 1,
        suite: str = "moptipy",
        f_name: str = F_NAME_RAW,
        f_standard: Optional[Dict[str, Union[int, float]]] = None) -> None:
    """
    Convert moptipy log data to IOHanalyzer log data.

    :param results_dir: the directory where we can find the results in moptipy
        format
    :param dest_dir: the directory where we would write the IOHanalyzer style
        data
    :param inst_name_to_func_id: convert the instance name to a function ID
    :param inst_name_to_dimension: convert an instance name to a function
        dimension
    :param inst_name_to_inst_id: convert the instance name an instance ID,
        which must be a positive integer number
    :param suite: the suite name
    :param f_name: the objective name
    :param f_standard: a dictionary mapping instances to standard values
    """
    source: Final[Path] = Path.directory(results_dir)
    dest: Final[Path] = Path.path(dest_dir)
    dest.ensure_dir_exists()
    logger(f"converting the moptipy log files in '{source}' to "
           f"IOHprofiler data in '{dest}'. First we load the data.")

    if (f_standard is not None) and (not isinstance(f_standard, dict)):
        raise type_error(f_standard, "f_standard", Dict)
    if not isinstance(suite, str):
        raise type_error(suite, "suite", str)
    if (len(suite) <= 0) or (" " in suite):
        raise ValueError(f"invalid suite name '{suite}'")
    if not callable(inst_name_to_func_id):
        raise type_error(
            inst_name_to_func_id, "inst_name_to_func_id", call=True)
    if not callable(inst_name_to_dimension):
        raise type_error(
            inst_name_to_dimension, "inst_name_to_dimension", call=True)
    if not callable(inst_name_to_inst_id):
        raise type_error(
            inst_name_to_inst_id, "inst_name_to_inst_id", call=True)

    # the data
    data: Final[Dict[str, Dict[str, Dict[int, List[
        Tuple[int, np.ndarray, np.ndarray]]]]]] = {}

    # this consumer collects all the data in a structured fashion
    def __consume(progress: Progress) -> None:
        nonlocal data
        nonlocal inst_name_to_func_id
        nonlocal inst_name_to_dimension
        nonlocal inst_name_to_inst_id

        _algo: Dict[str, Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]]
        if progress.algorithm in data:
            _algo = data[progress.algorithm]
        else:
            data[progress.algorithm] = _algo = {}
        _func_id: Final[str] = inst_name_to_func_id(progress.instance)
        if not isinstance(_func_id, str):
            raise type_error(_func_id, "function id", str)
        if (len(_func_id) <= 0) or ("_" in _func_id):
            raise ValueError(f"invalid function id '{_func_id}'.")
        _func: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]
        if _func_id in _algo:
            _func = _algo[_func_id]
        else:
            _algo[_func_id] = _func = {}
        _dim: Final[int] = inst_name_to_dimension(progress.instance)
        if not isinstance(_dim, int):
            raise type_error(_dim, "dim", int)
        if _dim <= 0:
            raise ValueError(f"invalid dimension: {_dim}.")
        _iid: Final[int] = inst_name_to_inst_id(progress.instance)
        if not isinstance(_iid, int):
            raise type_error(_iid, "instance id", int)
        if _iid <= 0:
            raise ValueError(f"invalid instance id: {_iid}.")
        _res: Final[Tuple[int, np.ndarray, np.ndarray]] = \
            (_iid, progress.time, progress.f)
        if _dim in _func:
            _func[_dim].append(_res)
        else:
            _func[_dim] = [_res]

    Progress.from_logs(source,
                       consumer=__consume,
                       time_unit=TIME_UNIT_FES,
                       f_name=check_f_name(f_name),
                       f_standard=f_standard,
                       only_improvements=True)

    if len(data) <= 0:
        raise ValueError("did not find any data!")
    logger(f"finished loading data from {len(data)} algorithms, "
           "now writing output.")

    for algo_name in sorted(data.keys()):
        algo = data[algo_name]
        algo_dir: Path = dest.resolve_inside(algo_name)
        algo_dir.ensure_dir_exists()
        logger(f"writing output for {len(algo)} functions of "
               f"algorithm '{algo_name}'.")
        for func_id in sorted(algo.keys()):
            func_dir: Path = algo_dir.resolve_inside(f"data_f{func_id}")
            func_dir.ensure_dir_exists()
            func = algo[func_id]
            logger(f"writing output for algorithm '{algo_name}' and "
                   f"function '{func_id}', got {len(func)} dimensions.")

            func_name = f"IOHprofiler_f{func_id}"
            with algo_dir.resolve_inside(
                    f"{func_name}.info").open_for_write() as info:
                for dimi in sorted(func.keys()):
                    dim_path = func_dir.resolve_inside(
                        f"{func_name}_DIM{dimi}.dat")
                    info.write(f"suite = '{suite}', funcId = '{func_id}', "
                               f"DIM = {dimi}, algId = '{algo_name}'\n")
                    info.write("%\n")
                    info.write(dim_path[len(algo_dir) + 1:])
                    with dim_path.open_for_write() as dat:
                        for per_dim in sorted(
                                func[dimi], key=lambda x:
                                (x[0], x[2][-1], x[1][-1])):
                            info.write(f", {per_dim[0]}:")
                            fes = per_dim[1]
                            f = per_dim[2]
                            info.write(num_to_str(fes[-1]))
                            info.write("|")
                            info.write(num_to_str(f[-1]))
                            dat.write(
                                '"function evaluation" "best-so-far f(x)"\n')
                            for i, ff in enumerate(f):
                                dat.write(f"{num_to_str(fes[i])} "
                                          f"{num_to_str(ff)}\n")
                        dat.write("\n")
                info.write("\n")
    del data
    logger("finished converting moptipy data to IOHprofiler data.")


# Run conversion if executed as script
if __name__ == '__main__':
    help_screen(
        "moptipy-to-IOHanalyzer data converter", __file__,
        "Convert log files obtained with moptipy to the "  # nosem
        "data format of the IOHprofiler (see "  # nosem
        "https://iohprofiler.github.io/IOHanalyzer/data/).",  # nosem
        [("source_dir", "the location of the moptipy data"),
         ("dest_dir", "the place to write the IOHprofiler data")])
    if len(sys.argv) != 3:
        raise ValueError("two command line arguments expected")
    moptipy_to_ioh_analyzer(sys.argv[1], sys.argv[2])
