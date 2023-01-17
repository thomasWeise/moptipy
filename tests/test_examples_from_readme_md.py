"""Test all the example code in the project's README.md file."""
import os.path
from typing import Final

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.temp import TempDir


def test_all_examples_from_readme_md() -> None:
    """Test all the example Python codes in the README.md file."""
    # First, we load the README.md file as a single string
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    readme = Path.file(base_dir.resolve_inside("README.md"))
    logger(f"executing all examples from README.md file {readme!r}.")
    text = readme.read_all_str()
    logger(f"got {len(text)} characters.")
    if len(text) <= 0:
        raise ValueError(f"README.md file at {readme!r} is empty?")
    del readme

    i2: int = -1
    # All examples start and end with ``` after a newline.
    mark1: Final[str] = "\n```"
    mark2: Final[str] = "python"  # python code starts with ```python

    wd: Final[str] = os.getcwd()  # get current working directory
    # We run all the example codes in a temporary directory.
    with TempDir.create() as td:  # create temporary working directory
        logger(f"using temp directory {td!r}.")
        os.chdir(td)  # set it as working directory
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
                logger(f"now processing fragment {logt}...")
                del logt

                logger("now compiling fragment.")
                code = compile(  # noqa # nosec
                    fragment, f"README.md:{i1}:{i2}",  # noqa # nosec
                    mode="exec")  # noqa # nosec
                logger("now executing fragment.")
                exec(code, {})  # execute file    # noqa # nosec
                logger("successfully executed example fragment.")

    os.chdir(wd)  # go back to current original directory
    logger("finished executing all examples from README.md.")
