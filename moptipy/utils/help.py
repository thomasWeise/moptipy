"""Print a help screen."""

import os.path
import sys
from typing import Final, Iterable, Tuple, List, Union

from moptipy.utils.console import logger
from moptipy.utils.path import _canonicalize_path
from moptipy.utils.types import type_error

#: The python interpreter in long form.
__INTERPRETER_LONG: Final[str] = _canonicalize_path(sys.executable)


def __get_python_interpreter_short() -> str:
    """
    Get the python interpreter.

    :returns: the fully-qualified path
    """
    bn = os.path.basename(__INTERPRETER_LONG)
    if bn.startswith("python3."):
        bn2 = bn[:7]
        interp2 = os.path.join(os.path.dirname(__INTERPRETER_LONG), bn2)
        if os.path.exists(interp2) and os.path.isfile(interp2):
            if _canonicalize_path(interp2) == __INTERPRETER_LONG:
                return bn2
    return bn


#: The python interpreter in short form.
__INTERPRETER_SHORT: Final[str] = __get_python_interpreter_short()
del __get_python_interpreter_short

#: the length of the base path
__BASE_PATH: Final[str] = _canonicalize_path(os.path.dirname(
    os.path.dirname(os.path.dirname(_canonicalize_path(__file__))))) + os.sep


def help_screen(title: str,
                file: str,
                text: str = "",
                args: Iterable[Union[Tuple[str, str],
                                     Tuple[str, str, bool]]] = ()) -> None:
    """
    Print a help screen on the console.

    :param title: the program title
    :param file: the calling python script
    :param text: the text explaining the program
    :param args: the command line arguments, as sequence of tuples
        `(title, description)` or `(title, description, optional)`
    :return: nothing
    """
    if not isinstance(title, str):
        raise type_error(title, "title", str)
    if not isinstance(file, str):
        raise type_error(file, "file", str)
    if not isinstance(text, str):
        raise type_error(text, "text", str)
    if not isinstance(args, Iterable):
        raise type_error(args, "args", Iterable)

    # get the module
    module: str = _canonicalize_path(file)
    end: int = len(module)
    start: int = 0
    if module.endswith(".py"):
        end -= 3
    if module.startswith(__BASE_PATH):
        start += len(__BASE_PATH)
    module = module[start:end].replace(os.sep, ".")
    mid: Final[str] = "' '"
    sargs: Final[str] = " ".join(
        f"[{t[0]}]" if (len(t) > 2) and t[2]  # type: ignore
        else t[0] for t in args)
    # prepare the text dump
    cons: Final[List[str]] = [
        title,
        f"usage: {__INTERPRETER_SHORT} -m {module} {sargs}",
        f" call: {__INTERPRETER_LONG} -m {module} '{mid.join(sys.argv[1:])}'"]

    if len(text) > 0:
        cons.append(text)

    # iterate over the arguments
    for arg in args:
        if not isinstance(arg, tuple):
            raise type_error(arg, "args[i]", Tuple)
        if not 1 < len(arg) < 4:
            raise ValueError(
                f"invalid argument tuple {arg}, should have length 2 or 3.")
        if not isinstance(arg[0], str):
            raise type_error(arg[0], "arg[0]", str)
        if len(arg[0]) <= 0:
            raise ValueError(f"invalid arg[0]: '{arg[0]}'")
        if not isinstance(arg[1], str):
            raise type_error(arg[1], "arg[1]", str)
        if len(arg[0]) <= 0:
            raise ValueError(f"invalid arg[1]: '{arg[1]}'")
        s = f"- {arg[0]}: {arg[1]}"
        if len(arg) > 2:
            if not isinstance(arg[2], bool):  # type: ignore
                raise type_error(arg[2],  # type: ignore
                                 "arg[2]", bool)
            if arg[2]:  # type: ignore
                s = f"{s} [optional]"
        cons.append(s)
    logger("\n".join(cons))
