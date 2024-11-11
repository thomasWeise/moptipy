"""
Example for bit-string based objective functions.

The problems noted here are mainly well-known benchmark problems from the
discrete optimization community. For these problems, we designed the dedicated
base class :class:`~moptipy.examples.bitstrings.bitstring_problem.\
BitStringProblem`.

The following benchmark problems are provided:

1.  :mod:`~moptipy.examples.bitstrings.ising1d`, the one-dimensional
    Ising model, where the goal is that all bits should have the same value as
    their neighbors in a ring.
2.  :mod:`~moptipy.examples.bitstrings.ising2d`, the two-dimensional
    Ising model, where the goal is that all bits should have the same value as
    their neighbors on a torus.
3.  The :mod:`~moptipy.examples.bitstrings.jump` problem is equivalent
    to :mod:`~moptipy.examples.bitstrings.onemax`, but has a deceptive
    region right before the optimum.
4.  The :mod:`~moptipy.examples.bitstrings.leadingones` problem,
    where the goal is to find a bit string with the maximum number of leading
    ones.
5.  The :mod:`~moptipy.examples.bitstrings.linearharmonic` problem,
    where the goal is to find a bit string with the all ones, like in
    :mod:`~moptipy.examples.bitstrings.onemax`, but this time all bits have
    a different weight (namely their index, starting at 1).
6.  The :mod:`~moptipy.examples.bitstrings.nqueens`, where the goal is to
    place `k` queens on a `k * k`-sized chess board such that no queen can
    beat any other queen.
7.  The :mod:`~moptipy.examples.bitstrings.onemax` problem, where the
    goal is to find a bit string with the maximum number of ones.
8.  The :mod:`~moptipy.examples.bitstrings.plateau` problem similar to the
    :mod:`~moptipy.examples.bitstrings.jump` problem, but this time the
    optimum is surrounded by a region of neutrality.
9.  The :mod:`~moptipy.examples.bitstrings.trap` problem, which is like
    OneMax, but with the optimum and worst-possible solution swapped. This
    problem is therefore highly deceptive.
10. The :mod:`~moptipy.examples.bitstrings.twomax` problem has the global
    optimum at the string of all `1` bits and a local optimum at the string
    of all `0` bits. Both have basins of attraction of about the same size.
11. The :mod:`~moptipy.examples.bitstrings.w_model`, a benchmark
    problem with tunable epistasis, uniform neutrality, and
    ruggedness/deceptiveness.
12. The :mod:`~moptipy.examples.bitstrings.zeromax` problem, where the
    goal is to find a bit string with the maximum number of zeros. This is the
    opposite of the OneMax problem.

Parts of the code here are related to the research work of
Mr. Jiazheng ZENG (曾嘉政), a Master's student at the Institute of Applied
Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at
Hefei University (合肥大学) in
Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
