"""Print a help screen."""


import argparse

from pycommons.io.arguments import make_argparser, make_epilog

from moptipy.version import __version__


def moptipy_argparser(file: str, description: str,
                      epilog: str) -> argparse.ArgumentParser:
    """
    Create an argument parser with default settings.

    :param file: the `__file__` special variable of the calling script
    :param description: the description string
    :param epilog: the epilogue string
    :returns: the argument parser

    >>> ap = moptipy_argparser(
    ...     __file__, "This is a test program.", "This is a test.")
    >>> isinstance(ap, argparse.ArgumentParser)
    True
    >>> "Copyright" in ap.epilog
    True
    """
    return make_argparser(
        file, description,
        make_epilog(epilog, 2022, 2024, "Thomas Weise",
                    url="https://thomasweise.github.io/moptipy",
                    email="tweise@hfuu.edu.cn, tweise@ustc.edu.cn"),
        __version__)
