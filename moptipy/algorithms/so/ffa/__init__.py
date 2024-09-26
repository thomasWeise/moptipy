"""
Algorithms and modules based on Frequency Fitness Assignment (FFA).

Frequency Fitness Assignment (FFA) is an algorithm plugin developed by the
team behind `moptipy`. FFA is suitable only for problems where the number
of different objective values is not too high. On such problems, which are
often discrete or combinatorial, it often can improve the result quality
that the optimization algorithms can reach, at the expense of requiring a
larger runtime.

1. Thomas Weise, Zhize Wu, Xinlu Li, and Yan Chen. Frequency Fitness
   Assignment: Making Optimization Algorithms Invariant under Bijective
   Transformations of the Objective Function Value. *IEEE Transactions on
   Evolutionary Computation* 25(2):307-319. April 2021. Preprint available at
   arXiv:2001.01416v5 [cs.NE] 15 Oct 2020.
   https://dx.doi.org/10.1109/TEVC.2020.3032090
2. Thomas Weise, Zhize Wu, Xinlu Li, Yan Chen, and Jörg Lässig. Frequency
   Fitness Assignment: Optimization without Bias for Good Solutions can be
   Efficient. *IEEE Transactions on Evolutionary Computation (TEVC)*. 2022.
   Early Access. https://dx.doi.org/10.1109/TEVC.2022.3191698
3. Thomas Weise, Mingxu Wan, Ke Tang, Pu Wang, Alexandre Devert, and Xin
   Yao. Frequency Fitness Assignment. *IEEE Transactions on Evolutionary
   Computation (IEEE-EC),* 18(2):226-243, April 2014.
   https://dx.doi.org/10.1109/TEVC.2013.2251885
4. Thomas Weise, Yan Chen, Xinlu Li, and Zhize Wu. Selecting a diverse set of
   benchmark instances from a tunable model problem for black-box discrete
   optimization algorithms. *Applied Soft Computing Journal (ASOC),*
   92:106269, June 2020. https://dx.doi.org/10.1016/j.asoc.2020.106269
5. Thomas Weise, Xinlu Li, Yan Chen, and Zhize Wu. Solving Job Shop Scheduling
   Problems Without Using a Bias for Good Solutions. In *Genetic and
   Evolutionary Computation Conference Companion (GECCO'21 Companion),*
   July 10-14, 2021, Lille, France. ACM, New York, NY, USA.
   ISBN 978-1-4503-8351-6. https://dx.doi.org/10.1145/3449726.3463124
6. Thomas Weise, Mingxu Wan, Ke Tang, and Xin Yao. Evolving Exact Integer
   Algorithms with Genetic Programming. In *Proceedings of the IEEE Congress
   on Evolutionary Computation (CEC'14), Proceedings of the 2014 World
   Congress on Computational Intelligence (WCCI'14),* pages 1816-1823,
   July 6-11, 2014, Beijing, China. Los Alamitos, CA, USA: IEEE Computer
   Society Press. ISBN: 978-1-4799-1488-3.
   https://dx.doi.org/10.1109/CEC.2014.6900292
7. Tianyu Liang, Zhize Wu, Jörg Lässig, Daan van den Berg, Sarah Louise
   Thomson, and Thomas Weise. Addressing the Traveling Salesperson Problem
   with Frequency Fitness Assignment and Hybrid Algorithms. *Soft Computing.*
   2024. https://dx.doi.org/10.1007/s00500-024-09718-8
"""
