# Release notes

## 0.2 series

### 0.2.0

Addition of a new bandit algorithm: OSSB. It comes with several implementations, 
depending on how the Graves-Lai problem is solved:

* Naïve exact solution, with the original formulation
* Exact solution, with an improved formulation and constraint generation
* Subgradient-based algorithm, with polynomial-time guarantees

The combinatorial problems have moved to a separate package, 
[Kombinator.jl](https://github.com/dourouc05/Kombinator.jl). The nonsmooth-
optimisation facilities required to implement OSSB in polynomial time were
short-lived in this package; they have already moved to 
[NonsmoothOptim.jl](NonsmoothOptim.jl).

Minor enhancements: 

* New functions `estimate_Δmax` and `estimate_Δmin`. 
* New function `maximum_solution_length`. 

## 0.1 series

### 0.1.4 (December 17, 2020)

ESCB2: propose an automatic choice of parameters. Refactor the way the
discretisation is performed to give more flexibility to the user. 

### 0.1.3 (February 17, 2020)

Update to JuMP 0.21. Fix an infinite loop in LP-based formulation for
elementary paths.

### 0.1.2 (February 7, 2020)

Quite a bit of bug fixing, especially regarding matching algorithms.

### 0.1.1 (January 31, 2020)

Update requirements for DataStructures to allow more versions.

Implement a refined approximation algorithm for budgeted maximum spanning trees.
It now provides a traditional approximation factor (1/2) instead of an
approximation term. See comments around `st_prim_budgeted_lagrangian_approx_half`
for more details.

### 0.1.0 (January 31, 2020)

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
