"""
Components for parsing and evaluating log files generated by experiments.

Via the :mod:`moptipy.api`, it is possible to log the progress or end
results of optimization algorithms runs in text-based log files.
With the methods in this package here, you can load and evaluate them. This
usually follows a multi-step approach: For example, you can first extract the
end results from several algorithms and instances into a single file via the
:class:`~moptipy.evaluation.end_results.EndResult`. This could then be
processed to per-algorithm or per-instance statistics using
:class:`~moptipy.evaluation.end_statistics.EndStatistics`.
"""


def __lang_setup():
    """Perform the internal language setup."""
    from moptipy.utils.lang import EN, DE, ZH  # pylint: disable=C0415
    import moptipy.api.logging as lg  # pylint: disable=C0415
    import moptipy.evaluation.base as bs  # pylint: disable=C0415

    EN.extend({
        "f": "f",
        "feasible": "feasible",
        "name": "name",
        "time": "time",
        "time_in_fes": "time in FEs",
        "time_in_ms": "time in ms",
        "lower_bound": "lower bound",
        "lower_bound_short": "lb",
        bs.F_NAME_NORMALIZED: "normalized f",
        bs.F_NAME_SCALED: "scaled f",
        bs.F_NAME_RAW: "f",
        bs.TIME_UNIT_FES: "time in FEs",
        bs.TIME_UNIT_MILLIS: "time in ms",
        lg.KEY_TOTAL_TIME_MILLIS: "total time in ms",
        lg.KEY_LAST_IMPROVEMENT_TIME_MILLIS: "last improvement time at ms",
        lg.KEY_TOTAL_FES: "total time in FEs",
        lg.KEY_LAST_IMPROVEMENT_FE: "last improvement at FE",
        "algorithm_on_instance": "algorithm \u00d7 instance",
        "ERT": "ERT",
        "ECDF": "ECDF",
        "setup": "setup",
        "best": "best",
        "worst": "worst",
        "summary": "summary"
    })

    DE.extend({
        "f": "f",
        "feasible": "realisierbar",
        "name": "Name",
        "time": "Zeit",
        "time_in_ms": "Zeit in ms",
        "time_in_fes": "Zeit in FEs",
        "lower_bound": "untere Schranke",
        "lower_bound_short": "us",
        bs.F_NAME_NORMALIZED: "normalisierte f",
        bs.F_NAME_SCALED: "skalierte f",
        bs.F_NAME_RAW: "f",
        bs.TIME_UNIT_FES: "Zeit in FEs",
        bs.TIME_UNIT_MILLIS: "Zeit in ms",
        lg.KEY_TOTAL_TIME_MILLIS: "Gesamtzeit in ms",
        lg.KEY_LAST_IMPROVEMENT_TIME_MILLIS: "letzte Verbesserung bei ms",
        lg.KEY_TOTAL_FES: "Gesamtzeit in FEs",
        lg.KEY_LAST_IMPROVEMENT_FE: "letzte Verbesserung bei FE",
        "algorithm_on_instance": "Algorithmus \u00d7 Instanz",
        "ERT": "ERT",
        "ECDF": "ECDF",
        "setup": "setup",
        "best": "beste",
        "worst": "schlechteste",
        "summary": "Übersicht"
    })

    ZH.extend({
        "f": "f",
        "feasible": "可行的",
        "name": "名称",
        "time": "时间",
        "time_in_ms": "时间(毫秒)",
        "time_in_fes": "时间(目标函数的评价)",
        "lower_bound": "下界",
        "lower_bound_short": "下界",
        bs.F_NAME_NORMALIZED: "归一化f",
        bs.F_NAME_SCALED: "标度f",
        bs.F_NAME_RAW: "f",
        bs.TIME_UNIT_FES: "时间(目标函数的评价)",
        bs.TIME_UNIT_MILLIS: "时间(毫秒)",
        lg.KEY_TOTAL_TIME_MILLIS: "总时间(毫秒)",
        lg.KEY_LAST_IMPROVEMENT_TIME_MILLIS: "最后一次改进是在(毫秒)",
        lg.KEY_TOTAL_FES: "总时间(目标函数的评价)",
        lg.KEY_LAST_IMPROVEMENT_FE: "最后一次改进是在(目标函数的评价)",
        "algorithm_on_instance": "优化算法 \u00d7 优化问题实例",
        "ERT": "经验估计运行时间",
        "ECDF": "经验累积分布函数",
        "setup": "算法配置",
        "best": "最好的",
        "worst": "最糟糕的",
        "summary": "总结"
    })


__lang_setup()  # invoke the language setup
del __lang_setup  # delete the language setup routine
