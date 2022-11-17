"""Test all the example code in the project's examples directory."""
import os.path
import subprocess  # nosec
import sys
from typing import Final

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.temp import TempDir


def test_examples_in_examples_directory():
    """Test all the examples in the examples directory."""
    # First, we resolve the directories
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    examples_dir = Path.directory(base_dir.resolve_inside("examples"))
    logger(f"executing all examples from examples directory '{examples_dir}'.")

    # we need to gather the python libraries and path to moptipy
    libpath: Final[str] = "PYTHONPATH"
    moptipy = Path.directory(base_dir.resolve_inside("moptipy"))
    sp = list(sys.path)
    if len(sp) <= 0:
        raise ValueError("Empty sys path?")
    sp.append(moptipy)
    ppp: Final[str] = os.pathsep.join(sp)
    del sp
    logger(f"new {libpath} is {ppp}.")

    with TempDir.create() as td:
        for name in os.listdir(examples_dir):
            if name.endswith(".py"):
                file: Path = Path.file(examples_dir.resolve_inside(name))
                cmd = [sys.executable, file]
                logger(f"now executing command {cmd} in dir '{td}'.")
                ret = subprocess.run(
                    cmd,  # nosec
                    check=True, text=True,  # nosec
                    timeout=500, cwd=td,  # nosec
                    env={libpath: ppp})  # nosec
                if ret is None:
                    raise ValueError("ret is None?")
                if ret.returncode != 0:
                    raise ValueError(f"return code is {ret.returncode}.")
                logger("successfully executed example.")

    logger("finished executing all examples from README.md.")
