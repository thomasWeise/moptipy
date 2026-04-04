"""Find the best per-instance files."""
import argparse
from contextlib import suppress
from os.path import getsize
from typing import Final

from pycommons.io.console import logger
from pycommons.io.csv import COMMENT_START
from pycommons.io.path import Path
from pycommons.strings.string_conv import num_to_str, str_to_num

from moptipy.api.logging import (
    FILE_SUFFIX,
    KEY_BEST_F,
    KEY_CURRENT_F,
    SECTION_FINAL_STATE,
    SECTION_RESULT_Y,
    SECTION_SETUP,
)
from moptipy.utils.help import moptipy_argparser
from moptipy.utils.logger import (
    KEY_VALUE_SEPARATOR,
    SECTION_END,
    SECTION_START,
)

#: the start tag for state
__START_STATE: Final[str] = f"{SECTION_START}{SECTION_FINAL_STATE}"
#: the start tag for state
__END_STATE: Final[str] = f"{SECTION_END}{SECTION_FINAL_STATE}"
#: the start tag for setup
__START_SETUP: Final[str] = f"{SECTION_START}{SECTION_SETUP}"
#: the end tag for setup
__END_SETUP: Final[str] = f"{SECTION_END}{SECTION_SETUP}"
#: the start tag for end result
__START_END_RESULT: Final[str] = f"{SECTION_START}{SECTION_RESULT_Y}"
#: the end tag for end result
__END_END_RESULT: Final[str] = f"{SECTION_END}{SECTION_RESULT_Y}"


def __find_best(path: Path, inst_key: str,
                best: dict[str, tuple[int | float, Path]]) -> None:
    """
    Recursively find the best per-instance files.

    :param path: the current path
    :param inst_key: the key of the instance
    :param best: the dictionary with the best results
    """
    if not path.exists():
        return
    if path.is_dir():
        logger(f"Entering directory {path!r}.")
        for f in path.list_dir():
            __find_best(f, inst_key, best)
        return
    if (not (path.is_file() and path.endswith(FILE_SUFFIX))) or (
            getsize(path) <= 0):
        return

    state: int = 0
    result: int = 0
    setup: int = 0
    best_f: int | float | None = None
    instance: str | None = None
    with path.open_for_read() as stream:
        for oline in stream:
            line = str.strip(oline)
            if line.startswith(COMMENT_START):
                continue
            if line.startswith(__START_STATE):
                if (state != 0) or (result == 1) or (setup == 1):
                    return
                state = 1
            elif line.startswith(__END_STATE):
                if state != 1:
                    return
                state = 2
                if (result == 2) and (setup == 2):
                    break
            elif line.startswith(__START_END_RESULT):
                if (result != 0) or (state == 1) or (setup == 1):
                    return
                result = 1
            elif line.startswith(__END_END_RESULT):
                if result != 1:
                    return
                result = 2
                if (state == 2) and (setup == 2):
                    break
            elif line.startswith(__START_SETUP):
                if (setup != 0) or (state == 1) or (result == 1):
                    return
                setup = 1
            elif line.startswith(__END_SETUP):
                if setup != 1:
                    return
                setup = 2
                if (state == 2) and (result == 2):
                    break
            elif (state == 1) and line.startswith(KEY_BEST_F):
                if best_f is not None:
                    continue
                text = line.split(KEY_VALUE_SEPARATOR)
                if list.__len__(text) != 2:
                    continue
                with suppress(ValueError):
                    best_f = str_to_num(text[1])
            elif (state == 1) and line.startswith(KEY_CURRENT_F):
                text = line.split(KEY_VALUE_SEPARATOR)
                if list.__len__(text) != 2:
                    continue
                with suppress(ValueError):
                    best_f = str_to_num(text[1])
            elif (setup == 1) and line.startswith(inst_key):
                text = line.split(KEY_VALUE_SEPARATOR)
                if list.__len__(text) != 2:
                    continue
                instance = str.strip(text[1])
                if str.__len__(instance) <= 0:
                    instance = None

    if (best_f is not None) and (instance is not None) and (result == 2) and (
            setup == 2) and (state == 2) and ((instance not in best) or (
            best_f < best[instance][0])):
        best[instance] = (best_f, path)


def find_per_instance_best(path: str, inst_key: str) -> list[tuple[
        str, int | float, Path]]:
    """
    Find the best results per instance.

    :param path: the path in which to search
    :param inst_key: the key of the instance, must appear in the
        SETUP section
    :return: the best results
    """
    inst_key = str.strip(inst_key)
    if str.__len__(inst_key) <= 0:
        raise ValueError("Invalid instance key.")
    best: Final[dict[str, tuple[int | float, Path]]] = {}
    __find_best(Path(path), inst_key, best)
    return sorted((inst, v[0], v[1]) for inst, v in best.items())


def make_per_instance_best_csv(source: str, dest: str, inst_key: str) -> None:
    """
    Create a CSV file with the best results per instance.

    :param source: the source file
    :param dest: the destination file
    :param inst_key: the key for finding the instance, must appear in the
        SETUP section
    """
    result: Final[list[tuple[str, int | float, Path]]] = (
        find_per_instance_best(source, inst_key))
    if list.__len__(result) <= 0:
        raise ValueError("Nothing found.")
    with Path(dest).open_for_write() as stream:
        stream.write("inst;f;path\n")
        for inst, f, path in result:
            stream.write(f"{inst};{num_to_str(f)};{path}\n")


# Run to parse all log files and to create csv
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__,
        "Make best CSV",
        ("Create a CSV file with the paths to the files with the"
         "best results on a per-instance basis."))
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the CSV file to be created",
        type=Path, nargs="?", default="./evaluation/best.txt")
    parser.add_argument(
        "instance_key", help="the key for the instance",
        type=str, nargs="?", default="f.i.name")

    args: Final[argparse.Namespace] = parser.parse_args()
    make_per_instance_best_csv(args.source, args.dest, args.instance_key)
