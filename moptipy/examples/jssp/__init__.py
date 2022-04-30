"""
The Job Shop Scheduling Problem is a good example for optimization tasks.

The JSSP is one of the most well-known combinatorial optimization tasks.
Here we provide a set of benchmark instances and examples for solving this
problem.
"""


def __lang_setup():
    """Perform the language setup."""
    from moptipy.utils.lang import DE, EN, ZH  # pylint: disable=C0415

    # the English language strings
    EN.extend({
        "gantt_info": "{gantt.instance.name} ({gantt.instance.jobs} "
                      "jobs \u00D7 {gantt.instance.machines} machines), "
                      "makespan {gantt[:,:,2].max()}",
        "gantt_info_no_ms": "{gantt.instance.name} ({gantt.instance.jobs} "
                            "jobs \u00D7 {gantt.instance.machines} machines)",
        "gantt_info_short": "{gantt.instance.name} / {gantt[:,:,2].max()}",
        "machine": "machine",
        "makespan": "makespan"
    })
    # the German language strings
    DE.extend({
        "gantt_info": "{gantt.instance.name} ({gantt.instance.jobs} "
                      "Jobs \u00D7 {gantt.instance.machines} Maschinen), "
                      "Makespan {gantt[:,:,2].max()}",
        "gantt_info_no_ms": "{gantt.instance.name} ({gantt.instance.jobs} "
                            "Jobs \u00D7 {gantt.instance.machines} Maschinen)",
        "gantt_info_short": "{gantt.instance.name} / {gantt[:,:,2].max()}",
        "machine": "Maschine",
        "makespan": "Makespan"
    })
    # the Chinese language strings
    ZH.extend({
        "gantt_info": "{gantt.instance.name}（{gantt.instance.jobs}份作业"
                      "\u00D7{gantt.instance.machines}台机器），"
                      "最大完工时间{gantt[:,:,2].max()}",
        "gantt_info_no_ms": "{gantt.instance.name}（{gantt.instance.jobs}份作业"
                            "\u00D7{gantt.instance.machines}台机器）",
        "gantt_info_short": "{gantt.instance.name} / {gantt[:,:,2].max()}",
        "machine": "机器",
        "makespan": "最大完工时间"
    })


__lang_setup()  # invoke the language setup
del __lang_setup  # delete the language setup routine
