"""Test all the example code in the project's README.md file."""
import os.path
import subprocess  # nosec
import sys
from typing import Final

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.temp import TempDir, TempFile


def test_all_examples_from_readme_md() -> None:
    """Test all the example Python codes in the README.md file."""
    # First, we load the README.md file as a single string
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    readme = Path.file(base_dir.resolve_inside("README.md"))
    logger(f"executing all examples from README.md file '{readme}'.")
    text = readme.read_all_str()
    logger(f"got {len(text)} characters.")
    if len(text) <= 0:
        raise ValueError(f"README.md file at '{readme}' is empty?")
    del readme

    i2: int = -1
    # All examples start and end with ``` after a newline.
    mark1: Final[str] = "\n```"
    mark2: Final[str] = "python"  # python code starts with ```python

    # we first need to gather the python libraries and path to moptipy
    libpath: Final[str] = "PYTHONPATH"
    moptipy = Path.directory(base_dir.resolve_inside("moptipy"))
    sp = list(sys.path)
    if len(sp) <= 0:
        raise ValueError("Empty sys path?")
    sp.append(moptipy)
    ppp: Final[str] = os.pathsep.join(sp)
    del sp
    logger(f"new {libpath} is {ppp}.")

    # We run all the example codes in a temporary directory.
    with TempDir.create() as td:
        logger(f"using temp dir '{td}'.")
        while True:
            # First, find the starting mark.
            i2 += 1
            i1 = text.find(mark1, i2)
            if i1 <= i2:
                break  # no starting mark anymore: done
            i1 += len(mark1)
            i2 = text.find(mark1, i1)
            if i2 <= i1:
                raise ValueError("No end mark for start mark?")

            fragment = text[i1:i2].strip()  # get the fragment
            if len(fragment) <= 0:
                raise ValueError("Empty fragment?")
            i2 += len(mark1)
            if fragment.startswith(mark2):  # it is a python fragment
                i3 = fragment.find("\n")
                if i3 < len(mark2):
                    raise ValueError("Did not find newline?")
                fragment = fragment[i3 + 1:].strip()
                if len(fragment) <= 0:
                    raise ValueError("Empty python fragment?")
                # OK, now we only have code left.
                logt = fragment[0:min(100, len(fragment))]\
                    .replace("\n", "\\n")
                logger(f"now executing fragment {logt}...")
                del logt

                with TempFile.create(suffix=".py", directory=td) as tf:
                    logger(f"using temp file '{tf}'.")
                    tf.write_all(fragment)

                    cmd = [sys.executable, os.path.basename(tf)]
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
