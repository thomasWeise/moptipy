"""
The Job Shop Scheduling Problem is a good example for optimization tasks.

The JSSP is one of the most well-known combinatorial optimization tasks.
Here we provide the well-known and often-used set of benchmark instances
(:mod:`~moptipy.examples.jssp.instance`) for this problem. Moreover, we
also run a fully-fledged example experiment applying many metaheuristics
to eight selected JSSP instances in module
:mod:`~moptipy.examples.jssp.experiment` and evaluate the data gathered from
this experiment in :mod:`~moptipy.examples.jssp.evaluation`. Be aware that
actually executing this experiment will take a very long time. However, you
can also find the complete experiment, every single algorithm used, and all
conclusions drawn with in-depth discussions in our online book "Optimization
Algorithms" at https://thomasweise.github.io/oa.

Anyway, here you have many JSSP benchmark instances, a useful search space,
and the common objective function. You can use them for your own research, if
you want: The JSSP benchmark instances can be loaded from a package-internal
resource by name using module :mod:`~moptipy.examples.jssp.instance`.
Solutions to a JSSP instances are :class:`~moptipy.examples.jssp.gantt.Gantt`
charts, i.e., diagrams which assign each operation of a job to a machine and
time slot. The :class:`~moptipy.examples.jssp.gantt_space.GanttSpace` is a
space operating on such charts and the
:class:`~moptipy.examples.jssp.makespan.Makespan` is the typical objective
function when solving JSSPs. Gantt charts can be encoded as permutations
with repetitions using the :mod:`~moptipy.examples.jssp.ob_encoding`.

1. Ronald Lewis Graham, Eugene Leighton Lawler, Jan Karel Lenstra, Alexander
   Hendrik George Rinnooy Kan. Optimization and Approximation in Deterministic
   Sequencing and Scheduling: A Survey. *Annals of Discrete Mathematics*
   5:287-326, 1979. doi: https://doi.org/10.1016/S0167-5060(08)70356-X.
   https://ir.cwi.nl/pub/18052/18052A.pdf.
2. Eugene Leighton Lawler, Jan Karel Lenstra, Alexander Hendrik George Rinnooy
   Kan, and David B. Shmoys. Sequencing and Scheduling: Algorithms and
   Complexity. Chapter 9 in Stephen C. Graves, Alexander Hendrik George
   Rinnooy Kan, and Paul H. Zipkin, editors, *Handbook of Operations Research
   and Management Science,* volume IV: *Production Planning and Inventory,*
   1993, pages 445-522. Amsterdam, The Netherlands: North-Holland Scientific
   Publishers Ltd. doi: https://doi.org/10.1016/S0927-0507(05)80189-6.
   http://alexandria.tue.nl/repository/books/339776.pdf.
3. Eugene Leighton Lawler. Recent Results in the Theory of Machine Scheduling.
   Chapter 8 in Achim Bachem, Bernhard Korte, and Martin Grötschel, editors,
   *Math Programming: The State of the Art,* 1982, pages 202-234.
   Bonn, Germany/New York, NY, USA: Springer-Verlag GmbH.
   ISBN: 978-3-642-68876-8. doi: https://doi.org/10.1007/978-3-642-68874-4_9.
4. Éric D. Taillard. Benchmarks for Basic Scheduling Problems. *European
   Journal of Operational Research (EJOR)* 64(2):278-285, January 1993.
   doi: https://doi.org/10.1016/0377-2217(93)90182-M.
   http://mistic.heig-vd.ch/taillard/articles.dir/Taillard1993EJOR.pdf.
5. Jacek Błażewicz, Wolfgang Domschke, and Erwin Pesch. The Job Shop
   Scheduling Problem: Conventional and New Solution Techniques. *European
   Journal of Operational Research (EJOR)* 93(1):1-33, August 1996.
   doi: https://doi.org/10.1016/0377-2217(95)00362-2.
6. Thomas Weise. *Optimization Algorithms.* 2021-2023. Hefei, Anhui, China:
   Institute of Applied Optimization, School of Artificial Intelligence and
   Big Data, Hefei University. https://thomasweise.github.io/oa
"""


def __lang_setup() -> None:
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
        "makespan": "makespan",
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
        "makespan": "Makespan",
    })
    # the Chinese language strings
    ZH.extend({
        "gantt_info": "{gantt.instance.name} ({gantt.instance.jobs}份作业"
                      "\u00D7{gantt.instance.machines}台机器), "
                      "最大完工时间{gantt[:,:,2].max()}",
        "gantt_info_no_ms":
            "{gantt.instance.name} ({gantt.instance.jobs}份作业"
            "\u00D7{gantt.instance.machines}台机器) ",
        "gantt_info_short": "{gantt.instance.name} / {gantt[:,:,2].max()}",
        "machine": "机器",
        "makespan": "最大完工时间",
    })


__lang_setup()  # invoke the language setup
del __lang_setup  # delete the language setup routine
