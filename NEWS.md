# Release notes

## 0.0.1

First release. Several combinatorial bandits algorithms are included:

* Thompson sampling
* LLR
* CUCB
* ESCB-2
* OLS-UCB

Only the last two algorithms provide state-of-the-art regret, but do not have
polynomial-time algorithm for all polynomial-time-solvable combinatorial
problems.

They can be applied on several combinatorial problems:

* perfect matching in bipartite graphs
* elementary paths (mostly, longest paths: all algorithms are guaranteed to work
  with DAGs, only LP-based algorithm with any graph)
* spanning trees
* m-sets

Several polynomial-time algorithms are implemented for ESCB-2:

* generic greedy heuristic
* exact, mathematical-optimisation-based algorithm (if the instance has a LP
  formulation of the problem: for now, all four basic instances)
* exact algorithm based on budgeted versions of the linear problems
  (provided for: m-sets, elementary paths, spanning trees)
