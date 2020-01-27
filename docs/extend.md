# How to extend this package?

There are two main ways of extending the functionality provided by this package:

* add a new type of combinatorial problem (or a solver for a given problem)
* add a new bandit algorithm

## State and trace

The state of the bandit is represented by the type `State{T}`. It retains
all the needed information for the bandit to take a new decision, like the
current round, the number of times each arm has been played, and the reward for
each arm.

A trace is the execution of a bandit algorithm. It is stored as an instance of
`Trace{T}`. It remembers the succession of states of this bandit, the arms
that were played, the obtained rewards, all details a policy may want to store,
and computation times.

## Adding a new combinatorial problem

The main task is to create a subtype of `CombinatorialInstance{T}`. There may be
several abstract types between `CombinatorialInstance{T}` and the concrete
problem type, depending on the code that may be shared by several problems.

An instance of a combinatorial problem should contain the parameters needed
for a simulation and a solver. The parameters most likely include the
probability distribution of reward for each arm.

The interface of `CombinatorialInstance{T}` is as follows.

* All implementers must provide the `n_arms` property. This integer should
  return the number of arms for this combinatorial problem instance, not the
  number of decisions to be taken at each round. For instance, for m-sets or
  knapsacks, it is the number of objects
* All implementers must provide the `optimal_average_reward` property.
  This floating-point number indicates the optimal reward a bandit can get
  in average. 
* These fields are used by the default implementations of some methods:
  * `solver`: the solver for this combinatorial problem. It is recommended
    to create a new abstract type dedicated to solvers of this combinatorial
    problem, so that they can easily be switched. The typical interface for
    solvers is described in the next section
  * `reward`: a mapping between arms and probability distributions of reward.
    Each probability distribution is supposed to be an instance of
    `Distribution`, as provided by Distributions.jl. If arms are simply
    numbered, this can be a vector of distributions (see m-sets);
    if arms are pairs of numbers (like edges in a graph), this can be a matrix
    of distributions (see perfect bipartite matchings). This type is given
    by `T`
  * `all_arm_indices`: for correlated arms only, a list of all arms for this
    instance, of type `T` (integers, tuples of integers). Their order must
    correspond to that of `reward`
* These methods are mandatory:
  * `initial_state`: returns an instance of `State{T}` with the right type,
    depending on how the arms are numbered (`Int` for simply numbered arms,
    like m-sets; `Tuple{Int, Int}` for double numbered arms, like perfect
    bipartite matchings)
* This package provides default implementations for:
  * `pull`: pulls the bandit with the given set of arms (represented as a
    vector). It returns the reward for each of the pulled arms (a vector whose
    indices correspond to the input set of arms), and the incurred regret
    (computed thanks to `solve_linear`)
  * `has_lp_formulation`: returns `false`, i.e. there is no LP formulation
    available for this problem using the current solver
  * `get_lp_formulation`: by default, throws an error if `has_lp_formulation`
    returns `false` and calls `has_lp_formulation` on the `solver` field.
    It returns a JuMP model (the underlying optimisation solver is supposed to
    be a field of the combinatorial solver)
  * `solve_linear`: by default, calls `solve_linear` on the `solver` field
  * `solve_budgeted_linear`: by default, calls `solve_budgeted_linear`
    on the `solver` field
  * `solve_all_budgeted_linear`: by default, calls `solve_all_budgeted_linear`
    on the `solver` field
* These methods are optional:
  * `initial_trace`: only required to start `simulate` with `with_trace=true`.
    Returns an instance of `Trace{T}` with the right type, similarly to
    `initial_state`. The default implementation should rarely be overridden
  * `is_feasible` and `is_partially_acceptable`: only used for the generic
    greedy algorithm. Determines whether the current set of arms can lead to
    a feasible solution (`is_partially_acceptable`) -- a feasible solution
    must be partially acceptable -- and whether the current set of arms is
    a feasible solution as is (`is_feasible`)

## Adding a new combinatorial solver

Each combinatorial solver is supposed to be a subtype of the corresponding
combinatorial instance solver type. A solver should contain all parameters
needed to compute the next solution the bandit should play (for instance,
the input graph, the number of objects to take, etc.). When asked to find a
solution, the solver will only be given the current state of the bandit
(i.e. elements from `State{T}`).

If needed, the combinatorial solver is supposed to have a handle on the
required low-level libraries that are used (such as a mathematical programming
solver or a dedicated library).

The combinatorial instance interface relies on the following methods, of which
only `solve_linear` is required:

* `has_lp_formulation`: returns `false`, i.e. this solver does not rely on an
  LP formulation
* `get_lp_formulation`: returns a JuMP model (the underlying optimisation
  solver is supposed to be a field of the combinatorial solver)
* `solve_linear`: solves the combinatorial problem with a linear objective
  function to maximise
* `solve_budgeted_linear`: solves the combinatorial problem with a linear
  objective function to maximise and a budget constraint (of the form
  ``a^T x \geq t``, `t` being a parameter given to the function). This budget
  is necessarily an integer
* `solve_all_budgeted_linear`: solves the combinatorial problem with a linear
  objective function to maximise and for all values of the budget until the
  given `max_budget`. All budgets are necessarily integers. This function
  should not be implemented if it does not bring any complexity advantage over
  repeated calls to `solve_budgeted_linear`

The parameters for the actual solving process, like the graph on which the
optimisation must take place, are not given to the constructor of the solver,
to limit the amount of code to be written. Rather, they are passed through
the `build!` method, called before any solving takes place.

## Adding a new bandit algorithm (a.k.a. policy)

A bandit algorithm must correspond to a concrete subtype of `Policy`. This
subtype is supposed to contain all needed parameters for the bandit algorithm
to work (like ε for ε-greedy approaches).

A policy might store running information in a `PolicyDetails` subtype.
This package only passes this data structure through to the user.

The main and only function of a policy is `choose_action`. The first argument
is the combinatorial instance to work on. The second one is used for multiple
dispatch and to choose the right policy. The third one is the current bandit
state (i.e. the `State{T}` object). A Boolean keyword argument `with_trace`
instructs the policy to store detailed runtime information in its
`PolicyDetails` object. This function is supposed to return a solution
(a vector of arms to play) and, if needed, a `PolicyDetails` object.
