# Portfolio Optimization with Trading Costs

In this investigation, we have compare the efficiencies of eight modelling package + solver combinations in solving Markowitz optimization models with none, linear, quadratic and regressed real-world trading cost models. `(Pyomo, IPOPT)` is our recommended open-sourced combination for its high efficiency, low variance in solving time and the ability to handle large-scale problems. The [report](final_report.pdf) is structured as follows: 

* **Section 1:** we begin with a description of the setup of the evaluation experiments. 
* **Section 2:** we present and interpret the results from the experiments. 
* **Section 3:** we provide extra details about the recommended IPOPT solver.
* **Section 4:** we discuss qualitatively some additional properties of the modelling packages.
* **Section 5:** we discuss about the limitations of this investigation and directions for future work.


### Performance Comparisons
![alt text](./__resources__/eval_eff.jpg?raw=true "Title")
**Figure 1:** The mean solving times (in log-scale) vs. the number of assets (with sample standard deviations represented by the shaded area). Legends are sorted in ascending order of the solving times. `MOSEK` (commercial) is the solver that consistently delivers superior performance for all types of trading costs. Among the open-sourced solvers, `(Pyomo, IPOTP)` and `(Pyomo, BONMIN)` have very good performance for complex non-linear constraints (i.e. quadratic and generic cost), and outperforms some commercial solvers including `GUROBI`.

![alt text](./__resources__/back_eff.jpg?raw=true "Title")
**Figure 2:** Boxplots of the solving times (in log-scale) of the modelling package + solver pairs. Legends are sorted in ascending order of solving times.
