"""Test all the example code in the project's examples directory."""
import os.path
from typing import Final

from numpy.random import default_rng

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.temp import TempDir


def test_examples_in_examples_directory() -> None:
    """Test all the examples in the examples directory."""
    # First, we resolve the directories
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    examples_dir = Path.directory(base_dir.resolve_inside("examples"))
    logger(
        f"executing all examples from examples directory {examples_dir!r}.")

    wd: Final[str] = os.getcwd()  # get current working directory
    with TempDir.create() as td:  # create temporary working directory
        logger(f"using temp directory {td!r}.")
        os.chdir(td)  # set it as working directory
        files: list[str] = os.listdir(examples_dir)
        logger(f"got {len(files)} potential files")
        default_rng().shuffle(files)  # shuffle the order for randomness
        for name in files:  # find all files in examples
            if name.endswith(".py"):  # if it is a python file
                file: Path = Path.file(examples_dir.resolve_inside(name))
                logger(f"now compiling file {file!r}.")
                code = compile(  # noqa # nosec
                    file.read_all_str(), file, mode="exec")  # noqa # nosec
                logger(f"now executing file {file!r}.")
                exec(code, {})  # execute file  # noqa # nosec
                logger(f"successfully executed example {file!r}.")
    os.chdir(wd)  # go back to current original directory
    logger("finished executing all examples from README.md.")
