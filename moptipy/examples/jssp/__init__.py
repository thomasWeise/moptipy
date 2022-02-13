"""
The Job Shop Scheduling Problem is a good example for optimization tasks.

The JSSP is one of the most well-known combinatorial optimization tasks.
Here we provide a set of benchmark instances and examples for solving this
problem.
"""
from moptipy.examples.jssp.gantt import Gantt
from moptipy.examples.jssp.gantt_space import GanttSpace
from moptipy.examples.jssp.instance import Instance
from moptipy.examples.jssp.makespan import Makespan
from moptipy.examples.jssp.ob_encoding import OperationBasedEncoding
from moptipy.evaluation.lang import Lang

__all__ = (
    "Gantt",
    "GanttSpace",
    "Instance",
    "Makespan",
    "OperationBasedEncoding")


# Below, we provide some standard language settings for our demo.

# the English language strings
Lang.get("en").extend({
    "gantt_info": "{gantt.instance.name} ({gantt.instance.jobs} "
                  "jobs \u00D7 {gantt.instance.machines} machines), "
                  "makespan {gantt[:,:,2].max()}",
    "gantt_info_no_ms": "{gantt.instance.name} ({gantt.instance.jobs} "
                        "jobs \u00D7 {gantt.instance.machines} machines)",
    "machine": "machine",
    "makespan": "makespan"
})
# the German language strings
Lang.get("de").extend({
    "gantt_info": "{gantt.instance.name} ({gantt.instance.jobs} "
                  "Jobs \u00D7 {gantt.instance.machines} Maschinen), "
                  "Makespan {gantt[:,:,2].max()}",
    "gantt_info_no_ms": "{gantt.instance.name} ({gantt.instance.jobs} "
                        "Jobs \u00D7 {gantt.instance.machines} Maschinen)",
    "machine": "Maschine",
    "makespan": "Makespan"
})
# the Chinese language strings
Lang.get("zh").extend({
    "gantt_info": "{gantt.instance.name}（{gantt.instance.jobs}份作业"
                  "\u00D7{gantt.instance.machines}台机器），"
                  "最大完工时间{gantt[:,:,2].max()}",
    "gantt_info_no_ms": "{gantt.instance.name}（{gantt.instance.jobs}份作业"
                        "\u00D7{gantt.instance.machines}台机器）",
    "machine": "机器",
    "makespan": "最大完工时间"
})
