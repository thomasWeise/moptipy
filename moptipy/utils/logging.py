"""Shared constants and functions for dealing with logs."""
import cmath
import math
from re import sub
from typing import List, Final

#: the file suffix to be used for log files
FILE_SUFFIX: Final[str] = ".txt"
#: the separator used in CSV files to separate columns
CSV_SEPARATOR: Final[str] = ";"
#: the character indicating the begin of a comment
COMMENT_CHAR: Final[str] = "#"
#: the character separating a scope prefix in a key-value section
SCOPE_SEPARATOR: Final[str] = "."
#: the indicator of the start of a log section
SECTION_START: Final[str] = "BEGIN_"
#: the indicator of the end of a log section
SECTION_END: Final[str] = "END_"
#: the replacement for "." in a file name
DECIMAL_DOT_REPLACEMENT: Final[str] = "d"
#: the separator of different filename parts
PART_SEPARATOR: Final[str] = "_"
#: the replacement for special characters
SPECIAL_CHAR_REPLACEMENT: Final[str] = "_"
#: the YAML-conform separator between a key and a value
KEY_VALUE_SEPARATOR: Final[str] = ": "
#: the hexadecimal version of a value
KEY_HEX_VALUE: Final[str] = "(hex)"


#: the key for algorithms
KEY_ALGORITHM: Final[str] = "algorithm"
#: the key for the instance
KEY_INSTANCE: Final[str] = "instance"

#: the progress csv section
SECTION_PROGRESS: Final[str] = "PROGRESS"
#: the FEs column for the progress CSV
PROGRESS_FES: Final[str] = "fes"
#: the time millis column for the progress CSV
PROGRESS_TIME_MILLIS: Final[str] = "timeMS"
#: the current objective value column for the progress CSV
PROGRESS_CURRENT_F: Final[str] = "f"

#: the end state
SECTION_FINAL_STATE: Final[str] = "STATE"
#: the total number of consumed FEs
KEY_TOTAL_FES: Final[str] = "totalFEs"
#: the total number of consumed milliseconds
KEY_TOTAL_TIME_MILLIS: Final[str] = "totalTimeMillis"
#: the best encountered objective value
KEY_BEST_F: Final[str] = "bestF"
#: the FE when the best objective value was reached
KEY_LAST_IMPROVEMENT_FE: Final[str] = "lastImprovementFE"
#: the time in milliseconds when the best objective value was reached
KEY_LAST_IMPROVEMENT_TIME_MILLIS: Final[str] = "lastImprovementTimeMillis"

#: the setup section
SECTION_SETUP: Final[str] = "SETUP"
#: the default log key for names of objects
KEY_NAME: Final[str] = "name"
#: the type of an object
KEY_TYPE: Final[str] = "type"
#: the inner type of an object
KEY_INNER_TYPE: Final[str] = "innerType"
#: the default log key for the lower bound of objective function values
KEY_F_LOWER_BOUND: Final[str] = "lowerBound"
#: the default log key for the upper bound of objective function values
KEY_F_UPPER_BOUND: Final[str] = "upperBound"
#: the maximum FEs of a black-box process
KEY_MAX_FES: Final[str] = "maxFEs"
#: the maximum runtime in milliseconds of a black-box process
KEY_MAX_TIME_MILLIS: Final[str] = "maxTimeMillis"
#: the goal objective value of a black-box process
KEY_GOAL_F: Final[str] = "goalF"
#: the random seed
KEY_RAND_SEED: Final[str] = "randSeed"
#: the random generator type
KEY_BBP_RAND_GENERATOR_TYPE: Final[str] = "randGenType"
#: the number of decision variables
KEY_SPACE_NUM_VARS: Final[str] = "nvars"

#: the scope of the solution space
SCOPE_SOLUTION_SPACE: Final[str] = "y"
#: the scope of the search space
SCOPE_SEARCH_SPACE: Final[str] = "x"
#: the scope of the objective function
SCOPE_OBJECTIVE_FUNCTION: Final[str] = "f"
#: the scope of the encoding
SCOPE_ENCODING: Final[str] = "g"
#: the scope of the optimization algorithm
SCOPE_ALGORITHM: Final[str] = "a"
#: the scope of the nullary search operator
SCOPE_OP0: Final[str] = "op0"
#: the scope of the unary search operator
SCOPE_OP1: Final[str] = "op1"
#: the scope of the binary search operator
SCOPE_OP2: Final[str] = "op2"

#: the automatically generated system info section
SECTION_SYS_INFO: Final[str] = "SYS_INFO"
#: information about the current session
SCOPE_SESSION: Final[str] = "session"
#: the time when the session was started
KEY_SESSION_START: Final[str] = "start"
#: the versions scope in the sys-info section
SCOPE_VERSIONS: Final[str] = "version"
#: the moptipy version key
KEY_MOPTIPY_VERSION: Final[str] = "moptipy"
#: the numpy version key
KEY_NUMPY_VERSION: Final[str] = "numpy"
#: the hardware scope in the sys-info section
SCOPE_HARDWARE: Final[str] = "hardware"
#: the number of CPUs
KEY_HW_N_CPUS: Final[str] = "n_cpus"
#: the key for the byte order
KEY_HW_BYTE_ORDER: Final[str] = "byteorder"
#: the key for the machine
KEY_HW_MACHINE: Final[str] = "machine"
#: the key for the cpu name
KEY_HW_CPU_NAME: Final[str] = "cpu"
#: the key for the memory size
KEY_HW_MEM_SIZE: Final[str] = "mem_size"
#: the operating system in the sys-info section
SCOPE_OS: Final[str] = "os"
#: the operating system name
KEY_OS_NAME: Final[str] = "name"
#: the operating system version
KEY_OS_VERSION: Final[str] = "version"
#: the python scope in the sys-info section
SCOPE_PYTHON: Final[str] = "python"
#: the python version
KEY_PYTHON_VERSION: Final[str] = "version"
#: the python implementation
KEY_PYTHON_IMPLEMENTATION: Final[str] = "implementation"

#: the resulting point in the solution space
SECTION_RESULT_Y: Final[str] = "RESULT_Y"
#: the resulting point in the search space
SECTION_RESULT_X: Final[str] = "RESULT_X"


def __recursive_replace(find: str, replace: str, src: str) -> str:
    """
    An internal function which performs a recursive replacement of strings.

    :param str find: the string to find
    :param str replace: the string with which it will be replaced
    :param str src: the string in which we search
    :return: the string src, with all occurrences of find replaced by replace
    :rtype: str
    """
    new_len = len(src)
    while True:
        src = src.replace(find, replace)
        old_len = new_len
        new_len = len(src)
        if new_len >= old_len:
            return src


def __replace_double(replace: str, src: str) -> str:
    return __recursive_replace(replace + replace, replace, src)


def sanitize_name(name: str) -> str:
    """
    Sanitize a name in such a way that it can be used as path component.

    >>> sanitize_name(" hello world ")
    'hello_world'
    >>> sanitize_name(" 56.6-455 ")
    '56d6-455'
    >>> sanitize_name(" _ i _ am _ funny   --6 _ ")
    'i_am_funny_-6'

    :param str name: the name that should be sanitized
    :return: the sanitized name
    :rtype: str
    :raises ValueError: if the name is invalid or empty
    :raises TypeError: if the name is None or not a string
    """
    if name is None:
        raise TypeError("Name string must not be None.")
    name = str(name)
    if not isinstance(name, str):
        raise TypeError("String representation of name must be instance "
                        f"of str, but is {type(name)}.")
    orig_name = name
    name = name.strip()
    name = __replace_double("-", name)
    name = __replace_double("_", name)
    name = __replace_double(".", name).replace(".", DECIMAL_DOT_REPLACEMENT)

    name = sub(r"[^\w\s-]", '', name)
    name = sub(r"\s+", PART_SEPARATOR, name)
    name = __replace_double("_", name)

    if name.startswith("_"):
        name = name[1:]

    if name.endswith("_"):
        name = name[:len(name) - 1]

    if len(name) <= 0:
        raise ValueError(
            f"Sanitized name must not become empty, but '{orig_name}' does.")

    return name


def sanitize_names(names: List[str]) -> str:
    """
    Sanitize a set of names.

    >>> sanitize_names(["", " sdf ", "", "5-3"])
    'sdf_5-3'
    >>> sanitize_names([" a ", " b", " c", "", "6", ""])
    'a_b_c_6'

    :param names: the list of names.
    :return: the sanitized name
    :retype: str
    """
    return PART_SEPARATOR.join([
        sanitize_name(name) for name in names if len(name) > 0])


def float_to_str(x: float) -> str:
    """
    Convert float to a string.

    :param float x: the floating point value
    :return: the string representation
    :rtype: str

    >>> float_to_str(1.3)
    '1.3'
    >>> float_to_str(1.0)
    '1'
    """
    if x == 0:
        return "0"
    s = repr(x)
    if math.isnan(x):
        raise ValueError(f"'{s}' not permitted.")
    if s.endswith(".0"):
        return s[:-2]
    return s


def complex_to_str(x: complex) -> str:
    """
    Convert complex number to a string.

    :param complex x: the complex floating point value
    :return: the string representation
    :rtype: str

    >>> complex_to_str(1.3+3j)
    '1.3+3j'
    >>> complex_to_str(1.0+0j)
    '1'
    >>> complex_to_str(1+0.2j)
    '1+0.2j'
    >>> complex_to_str(0+1j)
    '1j'
    >>> complex_to_str(0+0j)
    '0'
    >>> complex_to_str(-3j)
    '-3j'
    """
    if x == 0:
        return "0"
    y = abs(x)
    if y == x:
        return float_to_str(y)
    s = repr(x)
    if cmath.isnan(x):
        raise ValueError(f"'{s}' not permitted.")
    if s[0] == '(':
        s = s[1:-1]
    if s.endswith("+0j"):
        return s[:-3]
    if s.startswith("-0"):
        return s[(3 if s.startswith("-0+") else
                  2 if s.startswith("-0-") else 1):]
    return s


def bool_to_str(value: bool) -> str:
    """
    Convert a Boolean value to a string.

    :param bool value: the Boolean value
    :return: the string
    :rtype: str

    >>> print(bool_to_str(True))
    T
    >>> print(bool_to_str(False))
    F
    """
    return 'T' if value else 'F'


def str_to_bool(value: str) -> bool:
    """
    Convert a string to a boolean value.

    :param str value: the string value
    :return: the boolean value
    :rtype: bool

    >>> str_to_bool("T")
    True
    >>> str_to_bool("F")
    False
    >>> try:
    ...     str_to_bool("x")
    ... except ValueError as v:
    ...     print(v)
    Expected 'T' or 'F', but got 'x'.
    """
    if value == "T":
        return True
    if value == "F":
        return False
    raise ValueError(f"Expected 'T' or 'F', but got '{value}'.")
