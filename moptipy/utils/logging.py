from re import sub

from typing import List, Final, Union
from math import isfinite, isnan

#: the separator used in CSV files to separate columns
CSV_SEPARATOR: Final = ";"
#: the character indicating the begin of a comment
COMMENT_CHAR: Final = "#"
#: the character separating a scope prefix in a key-value section
SCOPE_SEPARATOR: Final = "."
#: the indicator of the start of a log section
SECTION_START: Final = "BEGIN_"
#: the indicator of the end of a log section
SECTION_END: Final = "END_"
#: the replacement for "." in a file name
DECIMAL_DOT_REPLACEMENT: Final = "d"
#: the separator of different filename parts
PART_SEPARATOR: Final = "_"
#: the replacement for special characters
SPECIAL_CHAR_REPLACEMENT: Final = "_"
#: the YAML-conform separator between a key and a value
KEY_VALUE_SEPARATOR: Final = ": "
#: the hexadecimal version of a value
KEY_HEX_VALUE: Final = "(hex)"

#: the default log key for names of objects
KEY_NAME: Final = "name"
#: the type of an object
KEY_TYPE: Final = "type"
#: the inner type of an object
KEY_INNER_TYPE: Final = "innerType"
#: the default log key for the lower bound of objective function values
KEY_F_LOWER_BOUND: Final = "lowerBound"
#: the default log key for the upper bound of objective function values
KEY_F_UPPER_BOUND: Final = "upperBound"
#: the maximum FEs of a black-box process
KEY_BBP_MAX_FES: Final = "maxFEs"
#: the maximum runtime in milliseconds of a black-box process
KEY_BBP_MAX_TIME_MILLIS: Final = "maxTimeMillis"
#: the goal objective value of a black-box process
KEY_BBP_GOAL_F: Final = "goalF"
#: the random seed
KEY_BBP_RAND_SEED: Final = "randSeed"
#: the random generator type
KEY_BBP_RAND_GENERATOR_TYPE: Final = "randGenType"
#: the total number of consumed FEs
KEY_ES_TOTAL_FES: Final = "totalFEs"
#: the total number of consumed milliseconds
KEY_ES_TOTAL_TIME_MILLIS: Final = "totalTimeMillis"
#: the best encountered objective value
KEY_ES_BEST_F: Final = "bestF"
#: the FE when the best objective value was reached
KEY_ES_LAST_IMPROVEMENT_FE: Final = "lastImprovementFE"
#: the time in milliseconds when the best objective value was reached
KEY_ES_LAST_IMPROVEMENT_TIME_MILLIS: Final = "lastImprovementTimeMillis"
#: the number of decision variables
KEY_SPACE_NUM_VARS: Final = "nvars"

#: the scope of the solution space
SCOPE_SOLUTION_SPACE: Final = "y"
#: the scope of the search space
SCOPE_SEARCH_SPACE: Final = "x"
#: the scope of the objective function
SCOPE_OBJECTIVE_FUNCTION: Final = "f"
#: the scope of the encoding
SCOPE_ENCODING: Final = "g"
#: the scope of the optimization algorithm
SCOPE_ALGORITHM: Final = "a"
#: the scope of the nullary search operator
SCOPE_OP0: Final = "op0"
#: the scope of the unary search operator
SCOPE_OP1: Final = "op1"
#: the scope of the binary search operator
SCOPE_OP2: Final = "op2"

#: the resulting point in the solution space
SECTION_RESULT_Y: Final = "RESULT_Y"
#: the resulting point in the search space
SECTION_RESULT_X: Final = "RESULT_X"
#: the end state
SECTION_FINAL_STATE: Final = "STATE"
#: the setup
SECTION_SETUP: Final = "SETUP"
#: the progress csv section
SECTION_PROGRESS: Final = "PROGRESS"

#: the FEs column for the progress CSV
PROGRESS_FES: Final = "fes"
#: the time millis column for the progress CSV
PROGRESS_TIME_MILLIS: Final = "timeMS"
#: the current objective value column for the progress CSV
PROGRESS_CURRENT_F: Final = "f"


def __recursive_replace(find: str, replace: str, src: str) -> str:
    """
    an internal function which performs a recursive replacement of strings

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
    sanitizes a name in such a way that it can be used as path component.

    >>> sanitize_name(" hello world ")
    'hello_world'
    >>> sanitize_name(" 56.6-455 ")
    '56d6-455'
    >>> sanitize_name(" _ i _ am _ funny   --6 _ ")
    'i_am_funny_-6'

    :param str name: the name that should be sanitized
    :return: the sanitized name
    :rtype: str
    :raises ValueError: if the name is either None or otherwise invalid
    """
    if name is None:
        raise ValueError("Name string must not be None.")
    name = str(name)
    if not isinstance(name, str):
        raise ValueError("String representation of name must be instance "
                         "of str, but is " + str(type(name)))
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
        raise ValueError("Sanitized name must not become empty, but '"
                         + orig_name + "' does.")

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


def format_float(x: Union[complex, float]) -> str:
    """
    Convert float-like to a string.
    :param Union[complex, float] x: the floating point value
    :return: the string representation
    >>> format_float(1.3)
    '1.3'
    >>> format_float(1.0)
    '1'
    """
    if x == 0:
        return "0"

    s = repr(x)
    if isfinite(x):
        if s.endswith(".0"):
            return s[:(len(s) - 2)]
        if s.endswith("+0j)"):
            return s[1:len(s) - 4]
        return s

    if isnan(x):
        ValueError("'" + s + "' not permitted.")

    return s
