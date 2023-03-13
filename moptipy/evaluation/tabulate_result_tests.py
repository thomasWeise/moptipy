r"""
Provides :func:`tabulate_result_tests` creating statistical comparison tables.

The function :func:`tabulate_result_tests` can compare two or more algorithms
on multiple problem instances by using the Mann-Whitney U test [1-3] with the
Bonferroni correction [4].

1. Daniel F. Bauer. Constructing Confidence Sets Using Rank Statistics.
   In *Journal of the American Statistical Association.* 67(339):687-690.
   September 1972. doi: https://doi.org/10.1080/01621459.1972.10481279.
2. Sidney Siegel and N. John Castellan Jr. *Nonparametric Statistics for The
   Behavioral Sciences.* 1988 In the Humanities/Social Sciences/Languages
   series. New York, NY, USA: McGraw-Hill. ISBN: 0-07-057357-3.
3. Myles Hollander and Douglas Alan Wolfe. *Nonparametric Statistical
   Methods.* 1973. New York, NY, USA: John Wiley and Sons Ltd.
   ISBN: 047140635X.
4. Olive Jean Dunn. Multiple Comparisons Among Means. In *Journal of the
   American Statistical Association.* 56(293):52-64. March 1961.
   doi: https://doi.org/10.1080/01621459.1961.10482090.
"""

from math import isfinite
from statistics import mean, median
from typing import Any, Callable, Final, Iterable, cast

from scipy.stats import mannwhitneyu  # type: ignore

from moptipy.api.logging import KEY_BEST_F
from moptipy.evaluation.end_results import EndResult
from moptipy.utils.formatted_string import FormattedStr
from moptipy.utils.markdown import Markdown
from moptipy.utils.number_renderer import (
    DEFAULT_NUMBER_RENDERER,
    NumberRenderer,
)
from moptipy.utils.path import Path
from moptipy.utils.table import Table
from moptipy.utils.text_format import TextFormatDriver
from moptipy.utils.types import type_error

#: the string constant for alpha
__ALPHA: Final[FormattedStr] = FormattedStr.special("\u03b1")
#: the string constant for dash or unclear
__DASH: Final[FormattedStr] = FormattedStr.special("\u2014")
#: the character for less
__LESS: Final[FormattedStr] = FormattedStr("<", code=True)
#: the character for equal
__UNCLEAR: Final[FormattedStr] = FormattedStr("?", code=True)
#: the character for greater
__GREATER: Final[FormattedStr] = FormattedStr(">", code=True)

#: the name of the win-equal-loss column
__WEL: Final[tuple[FormattedStr, str, FormattedStr, str, FormattedStr]] \
    = __LESS, " / ", __UNCLEAR, " / ", __GREATER


def __compare(a: list[int | float],
              b: list[int | float]) -> int:
    """
    Compare two lists of numbers.

    `-1` is returned if `a` has both a smaller mean and median value than `b`.
    `1` is returned if `a` has both a larger mean and median value than `b`.
    `0` is returned otherwise.

    :param a: the first list of numbers
    :param b: the second list of numbers
    :returns: `-1` if the numbers in `a` tend to be smaller than those in `b`,
        `1` if the numbers in `a` tend to be larger than those in `b`,
        `0` if undecided
    """
    smaller: bool
    ma = mean(a)
    mb = mean(b)
    if ma < mb:
        smaller = True
    elif ma > mb:
        smaller = False
    else:
        return 0
    ma = median(a)
    mb = median(b)
    if ma < mb:
        return -1 if smaller else 0
    if ma > mb:
        return 0 if smaller else 1
    return 0


def __compare_to_str(cmp: int, p_str: FormattedStr | None,
                     p: float, alpha_prime: float) -> str | list[str]:
    """
    Transform a comparison result and a string to an output.

    :param cmp: the comparison result
    :param p_str: the rendered `p` value string
    :param p: the actual `p` value
    :param alpha_prime: the `alpha` value
    :returns: the output
    """
    if cmp == 0:
        return __DASH
    if p >= alpha_prime:
        return [p_str, " ", __UNCLEAR]
    if cmp < 0:
        return [p_str, " ", __LESS]
    if cmp > 0:
        return [p_str, " ", __GREATER]
    raise ValueError(
        f"huh? p={p}, p_str={p_str}, cmp={cmp}, alpha'={alpha_prime}")


def tabulate_result_tests(
        end_results: Iterable[EndResult],
        file_name: str = "tests",
        dir_name: str = ".",
        alpha: float = 0.02,
        text_format_driver: TextFormatDriver | Callable[[], TextFormatDriver]
        = Markdown.instance,
        algorithm_sort_key: Callable[[str], Any] = lambda a: a,
        instance_sort_key: Callable[[str], Any] = lambda i: i,
        instance_namer: Callable[[str], str] = lambda x: x,
        algorithm_namer: Callable[[str], str] = lambda x: x,
        use_lang: bool = False,
        p_renderer: NumberRenderer = DEFAULT_NUMBER_RENDERER,
        value_getter: Callable[[EndResult], int | float]
        = EndResult.getter(KEY_BEST_F)) -> Path:
    r"""
    Tabulate the results of statistical comparisons of end result qualities.

    `end_results` contains a sequence of
    :class:`~moptipy.evaluation.end_results.EndResult` records, each of which
    represents the result of one run of one algorithm on one instance. This
    function performs a two-tailed Mann-Whitney U test for each algorithm pair
    on each problem instance to see if the performances are statistically
    significantly different. The results of these tests are tabulated,
    together with their `p`-values, i.e., the probabilities that the observed
    differences would occur if the two algorithms would perform the same.

    If `p` is sufficiently small, this means that it is unlikely that the
    difference in performance of the two compared algorithms that was observed
    stems from randomness. But what does "sufficiently small" mean?
    As parameter, this function accepts a significance threshold
    `0<alpha<0.5`. `alpha` is, so to say, the upper limit of the "probability
    to be wrong" if we claim something like "algorithm A is better than
    algorithm B" that we are going to accept. In other words, if the table
    says that algorithm A is better than algorithm B, the chance that this is
    wrong is not more than `alpha`.

    However, if we do many such tests, our chance to make at least one mistake
    grows. If we do `n_tests` tests, then the chance that all of them are
    "right" would be `1-[(1-alpha)^n_tests]`. Since we are going to do
    multiple tests, the Bonferroni correction is therefore applied and
    `alpha'=alpha/n_tests` is computed. Then, the chance to have at least one
    of the `n_tests` test results to be wrong is not higher than `alpha`.

    The test results are presented as follows:
    The first column of the generated table denotes the problem instances.
    Each of the other columns represents a pair of algorithms. In each cell,
    the pair is compared based on the results on the instance of the row. The
    cell ten holds the `p`-value of the two-tailed Mann-Whitney U test. If the
    first algorithm is significantly better (at `p<alpha'`) than the second
    algorithm, then the cell is marked with `<`. If the first algorithm is
    significantly worse (at `p<alpha'`) than the second algorithm, then the
    cell is marked with `>`. If the observed differences are not significant
    (`p>=alpha'`), then the cell is marked with `?`.

    However, there could also be a situation where a statistical comparison
    makes no sense as no difference could reliably be detected anyway. For
    example, if one algorithm has a smaller median result but a larger mean
    result, or if the medians are the same, or if the means are the same.
    Regardless of what outcome a test would have, we could not really claim
    that any of the algorithms was better or worse. In such cases, no test
    is performed and `-` is printed instead (signified by `&mdash;` in the
    markdown format).

    Finally, the bottom row sums up the numbers of `<`, `?`, and `>` outcomes
    for each algorithm pair.

    Depending on the parameter `text_format_driver`, the tables can be
    rendered in different formats, such as
    :py:class:`~moptipy.utils.markdown.Markdown`,
    :py:class:`~moptipy.utils.latex.LaTeX`, and
    :py:class:`~moptipy.utils.html.HTML`.

    :param end_results: the end results data
    :param file_name: the base file name
    :param dir_name: the base directory
    :param alpha: the threshold at which the two-tailed test result is
        accepted.
    :param text_format_driver: the text format driver
    :param algorithm_sort_key: a function returning sort keys for algorithms
    :param instance_sort_key: a function returning sort keys for instances
    :param instance_namer: the name function for instances receives an
        instance ID and returns an instance name; default=identity function
    :param algorithm_namer: the name function for algorithms receives an
        algorithm ID and returns an instance name; default=identity function
    :param use_lang: should we use the language to define the filename
    :param p_renderer: the renderer for all probabilities
    :param value_getter: the getter for the values that should be compared. By
        default, the best obtained objective values are compared. However, if
        you let the runs continue until they reach a certain goal quality,
        then you may want to compare the runtimes consumed until that quality
        is reached. Basically, you can use any of the getters provided by
        :meth:`moptipy.evaluation.end_results.EndResult.getter`, but you must
        take care that the comparison makes sense, i.e., compare qualities
        under fixed-budget scenarios (the default behavior) or compare
        runtimes under scenarios with goal qualities - but do not mix up the
        scenarios.
    :returns: the path to the file with the tabulated test results
    """
    # Before doing anything, let's do some type checking on the parameters.
    # I want to ensure that this function is called correctly before we begin
    # to actually process the data. It is better to fail early than to deliver
    # some incorrect results.
    if not isinstance(end_results, Iterable):
        raise type_error(end_results, "end_results", Iterable)
    if not isinstance(file_name, str):
        raise type_error(file_name, "file_name", str)
    if not isinstance(dir_name, str):
        raise type_error(dir_name, "dir_name", str)
    if not isinstance(alpha, float):
        raise type_error(alpha, "alpha", float)
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"invalid alpha={alpha}, must be 0<alpha<0.5.")
    if callable(text_format_driver):
        text_format_driver = text_format_driver()
    if not isinstance(text_format_driver, TextFormatDriver):
        raise type_error(text_format_driver, "text_format_driver",
                         TextFormatDriver, True)
    if not callable(algorithm_namer):
        raise type_error(algorithm_namer, "algorithm_namer", call=True)
    if not callable(instance_namer):
        raise type_error(instance_namer, "instance_namer", call=True)
    if not isinstance(use_lang, bool):
        raise type_error(use_lang, "use_lang", bool)

    if not isinstance(p_renderer, NumberRenderer):
        raise type_error(p_renderer, "p_renderer", NumberRenderer)
    if not callable(value_getter):
        raise type_error(value_getter, "value_getter", call=True)

    # now gather the data: algorithms -> instances -> bestF
    data_dict: dict[str, dict[str, list[int | float]]] = {}
    inst_set: set[str] = set()
    for i, end_result in enumerate(end_results):
        if not isinstance(end_result, EndResult):
            raise type_error(end_result, f"end_results[{i}]", EndResult)
        value = value_getter(end_result)
        if not isinstance(value, int | float):
            raise type_error(
                value, f"value_getter({end_result}", (int, float))
        if not isfinite(value):
            raise ValueError(f"value_getter({end_result}={value}")
        inst_set.add(end_result.instance)
        if end_result.algorithm in data_dict:
            inst_res = data_dict[end_result.algorithm]
        else:
            data_dict[end_result.algorithm] = inst_res = {}
        if end_result.instance in inst_res:
            inst_res[end_result.instance].append(value)
        else:
            inst_res[end_result.instance] = [value]

    # finished collecting data
    n_algos: Final[int] = len(data_dict)
    if n_algos <= 0:
        raise ValueError("end results are empty!")
    if not (1 < n_algos < 5):
        raise ValueError(f"invalid number {n_algos} of "
                         "algorithms, must be 1<n_algos<5.")
    instances: Final[list] = sorted(inst_set, key=instance_sort_key)
    n_insts: Final[int] = len(instances)
    algorithms: Final[list] = sorted(data_dict.keys(), key=algorithm_sort_key)

    # validate data
    for algo in algorithms:
        if len(data_dict[algo]) != n_insts:
            raise ValueError(
                f"algorithm {algo!r} has only data for {len(data_dict[algo])}"
                f" instances, but should have for {n_insts}.")

    # compute the Bonferroni correction
    n_cols: Final[int] = ((n_algos - 1) * n_algos) // 2
    n_pairs: Final[int] = n_insts * n_cols
    alpha_prime: Final[float] = alpha / n_pairs
    if not (0.0 < alpha_prime < 0.5):
        raise ValueError(f"invalid alpha'={alpha_prime}, must be 0<alpha'"
                         f"<0.5, stems from alpha={alpha} and N={n_pairs}")

    # compute the comparison results
    comparisons: Final[list[list[int]]] = \
        [[__compare(data_dict[algorithms[i]][inst],
                    data_dict[algorithms[j]][inst])
          for inst in instances]
         for i in range(n_algos - 1)
         for j in range(i + 1, n_algos)]

    # compute the p_values
    p_values: Final[list[list[float]]] = \
        [[float(mannwhitneyu(data_dict[algorithms[ij[0]]][inst],
                             data_dict[algorithms[ij[1]]][inst]).pvalue)
          if comparisons[z][k] != 0 else None
          for k, inst in enumerate(instances)]
         for z, ij in enumerate([(ii, jj) for ii in range(n_algos - 1)
                                 for jj in range(ii + 1, n_algos)])]

    # now format all p values to strings
    p_flat: Final[list[float]] = [y for x in p_values for y in x]
    p_flat.append(alpha)
    p_flat.append(alpha_prime)
    p_flat_strs: Final[list[FormattedStr | None]] = \
        p_renderer.render(p_flat)
    alpha_str: FormattedStr = cast(FormattedStr, p_flat_strs[-2])
    a2 = FormattedStr.number(alpha)
    if len(a2) <= len(alpha_str):
        alpha_str = a2
    alpha_prime_str: FormattedStr = cast(FormattedStr, p_flat_strs[-1])
    a2 = FormattedStr.number(alpha_prime)
    if len(a2) <= len(alpha_prime_str):
        alpha_prime_str = a2
    del p_flat_strs[-1]
    del p_flat_strs[-1]
    p_strs: Final[list[list[FormattedStr | None]]] = \
        [p_flat_strs[(i * n_insts):((i + 1) * n_insts)] for i in range(n_cols)]

    summary_row: list[Iterable[str] | str] | None = None
    if n_insts > 0:
        summary_row = [__WEL]
        for i, col in enumerate(comparisons):
            wins: int = 0
            equals: int = 0
            losses: int = 0
            pv: list[float] = p_values[i]
            for j, cv in enumerate(col):
                if (cv == 0) or (pv[j] >= alpha_prime):
                    equals += 1
                elif cv < 0:
                    wins += 1
                else:
                    losses += 1
            summary_row.append(f"{wins}/{equals}/{losses}")

    #: the columns
    cols: Final[list[list[str | list[str]]]] = \
        [[__compare_to_str(comparisons[i][j], p_strs[i][j], p_values[i][j],
                           alpha_prime)
          for j in range(n_insts)]
         for i in range(n_cols)]
    cols.insert(0, [FormattedStr.add_format(instance_namer(inst), code=True)
                    for inst in instances])

    # We now got the p_values, the comparison results, and the p_strs.

    # We now can render the headlines. If there is only one pair of
    # algorithms, we render a single headline with all the information.
    # If there are multiple algorithm pairs, we do two headlines.
    head_lines: list[list[str | list[str]]]
    multi_header: Final[bool] = \
        (n_algos > 2) and (not isinstance(text_format_driver, Markdown))
    if multi_header:
        head_lines = [[], []]

        def __app(a, b) -> None:
            nonlocal head_lines
            head_lines[0].append(a)
            head_lines[1].append(b)
    else:
        head_lines = [[]]

        def __app(a, b) -> None:
            nonlocal head_lines
            x = []
            if isinstance(a, str):
                x.append(a)
            else:
                x.extend(a)
            x.append(" ")
            if isinstance(b, str):
                x.append(b)
            else:
                x.extend(b)
            head_lines[0].append(x)

    __app("Mann-Whitney U", [__ALPHA, "=", alpha_str, ", ", __ALPHA,
                             "'=", alpha_prime_str])
    for i in range(n_algos - 1):
        for j in range(i + 1, n_algos):
            an1 = FormattedStr.add_format(algorithm_namer(algorithms[i]),
                                          code=True)
            an2 = FormattedStr.add_format(algorithm_namer(algorithms[j]),
                                          code=True)
            if len(an1) >= len(an2):
                __app(an1, ["vs. ", an2])
            else:
                __app([an1, " vs."], an2)

    # write the table
    dest: Final[Path] = text_format_driver.filename(
        file_name, dir_name, use_lang)
    with dest.open_for_write() as wd, Table(
            wd, ("l" if multi_header else "r") + ("c" * n_cols),
            text_format_driver) as table:
        with table.header() as header:
            for hl in head_lines:
                with header.row() as row:
                    for cell in hl:
                        row.cell(cell)
        with table.section() as section:
            section.cols(cols)
        if summary_row is not None:
            with table.section() as section, section.row() as row:
                for s in summary_row:
                    row.cell(s)

    return dest
