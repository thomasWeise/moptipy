"""Utilities used in other moptipy modules."""

from typing import Final

import moptipy.version
from moptipy.utils.cache import is_new
from moptipy.utils.io import canonicalize_path, enforce_dir, enforce_file, \
    file_create_or_fail, file_create_or_truncate, file_ensure_exists, \
    TempDir, TempFile
from moptipy.utils.logger import CsvSection, FileLogger, InMemoryLogger, \
    KeyValueSection, Logger, TextSection
from moptipy.utils.logging import float_to_str, sanitize_name, \
    sanitize_names, num_to_str
from moptipy.utils.nputils import int_range_to_dtype, rand_seed_check, \
    rand_seed_generate, rand_generator, rand_seeds_from_str

__version__: Final[str] = moptipy.version.__version__

__all__ = (
    "canonicalize_path",
    "CsvSection",
    "enforce_dir",
    "enforce_file",
    "file_create_or_fail",
    "file_create_or_truncate",
    "file_ensure_exists",
    "FileLogger",
    "float_to_str",
    "InMemoryLogger",
    "int_range_to_dtype",
    "is_new",
    "KeyValueSection",
    "Logger",
    "num_to_str",
    "rand_generator",
    "rand_seed_check",
    "rand_seed_generate",
    "rand_seeds_from_str",
    "sanitize_name",
    "sanitize_names",
    "TempDir",
    "TempFile",
    "TextSection")
