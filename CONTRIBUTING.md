# Contributing to moptipy

Thank you for supporting the [`moptipy`](https://thomasweise.github.io/moptipy), the Metaheuristic Optimization in Python package.
`moptipy` is a library for implementing, using, and experimenting with metaheuristic optimization algorithms.
Our project is developed for scientific, educational, and industrial applications.


## 1. Contributing

One of the pillars of our project is extensive documentation, the implementation of many style-guides, the use of unit tests, and the use of many static code analysis tools.
Our [make build](https://thomasweise.github.io/moptipy/Makefile.html) requires the code to pass the checks of more than [20 tools](https://thomasweise.github.io/moptipy/index.html#unit-tests-and-static-analysis).
This may make it complicated to submit code contributions via git pull requests.

The preferred way to contribute to this project therefore is by opening [issues](https://github.com/thomasWeise/moptipy/issues).

If you nevertheless submit a git pull or otherwise code-based contribution, then it should ideally pass all these checks.
In other words, the [make build](https://thomasweise.github.io/moptipy/Makefile.html) requires the code to pass the checks of more than [20 tools](https://thomasweise.github.io/moptipy/index.html#unit-tests-and-static-analysis) should succeed on your local system.

If that is not possible, you can still submit the pull request or otherwise code-based contribution.
However, we will then need to invest more work to manually check to see how and whether it can be integrated into our code base.
On the one hand, this may take quite some time.
On the other hand, it may also mean that we eventually reject the request and manually integrate a very modified version of the code -- but we would of course give proper credit.
Finally, it could also turn out that we simply cannot integrate the contribution at the current time.
We will, however, definitely try our best.

We believe that, in the long run, having very clearly documented code that follows best practices and is thoroughly tested wherever possible will benefit the value of our project.
The downside is that it takes a lot of resources, time, and nit-picking energy.

If your contribution concerns the *security* of `moptipy`, please consider our [security policy](https://thomasweise.github.io/moptipy/SECURITY.html).


## 2. License

`moptipy` is provided to the public as open source software under the [GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007](https://thomasweise.github.io/moptipy/LICENSE.html).
Terms for other licenses, e.g., for specific industrial applications, can be negotiated with Dr. Thomas Weise (who can be reached via the [contact information](#3-contact) below).
Dr. Thomas Weise holds the copyright of this package *except* for the JSSP instance data in file [`moptipy/examples/jssp/instances.txt`](https://github.com/thomasWeise/moptipy/blob/main/moptipy/examples/jssp/instances.txt).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in `moptipy` by you shall be under the above license, terms, and conditions, without any additional terms or conditions.
Unless explicitly stating otherwise, contributors accept that their contributions will be licensed under the project terms.
This also means that they grant Dr. Thomas Weise non-exclusive copyright of their contributions.


## 3. Contact

If you have any questions or suggestions, please contact
Prof. Dr. [Thomas Weise](http://iao.hfuu.edu.cn/5) (汤卫思教授) of the 
Institute of Applied Optimization (应用优化研究所, [IAO](http://iao.hfuu.edu.cn)) of the
School of Artificial Intelligence and Big Data ([人工智能与大数据学院](http://www.hfuu.edu.cn/aibd/)) at
[Hefei University](http://www.hfuu.edu.cn/english/) ([合肥学院](http://www.hfuu.edu.cn/)) in
Hefei, Anhui, China (中国安徽省合肥市) via
email to [tweise@hfuu.edu.cn](mailto:tweise@hfuu.edu.cn) with CC to [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn).
