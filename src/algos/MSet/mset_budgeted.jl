"""
An instance of the budgeted m-set problem. The budget is a minimum total
weight that must be taken for a solution to be feasible.

All weights are supposed to be positive integers, the budget too (this
hypothesis is crucial to get efficient algorithms). Values are not restricted.

It can be formalised as follows:

``\\max \\sum_i \\mathrm{values}_i x_i``
``\\mathrm{s.t.} \\sum_i x_i \\leq m, \\quad \\sum_i \\mathrm{weights}_i x_i \\geq \\mathrm{budget}, \\qquad x \\in \\{0, 1\\}^d``

`max_weight` is a parameter that may be used by the algorithm and corresponds
to the largest value a single weight may take. If not present, it is computed
as `max(weights)`.

`budget` can be ignored; in that case, it will be computed as the maximum
budget that may have a feasible solution, i.e. `d * max_weight`, `d` being
the dimension of the problem (the number of objects). If a `budget` is given
and the algorithm computes all solutions for lower budgets, then the algorithm
is allowed to stop at the given budget (i.e. it will not be possible to
retrieve a solution for a larger budget).
"""
struct BudgetedMSetInstance
  values::Vector{Float64}
  weights::Vector{Int}
  m::Int
  budget::Int # weights * solution >= budget
  max_weight::Int

  function BudgetedMSetInstance(values::Vector{Float64}, weights::Vector{Int}, m::Int;
                                budget::Union{Nothing, Int}=nothing,
                                max_weight::Union{Nothing, Int}=nothing)
    # Error checking.
    if m < 0
      error("m is less than zero: there is no solution.")
    end

    if m == 0
      error("m is zero: the only solution is to take no items, if the budget constraint is satisfied.")
    end

    if ! isnothing(budget) && budget < 0
      error("Budget is present and negative.")
    end

    if any(weights .< 0)
      error("At least a weight is negative.")
    end

    if length(values) != length(weights)
      error("Not the same number of values and weights; these two vectors must have the same size.")
    end

    # Complete the optional parameters.
    if isnothing(max_weight)
      max_weight = maximum(weights)
    end

    if isnothing(budget)
      d = length(values) # Dimension of the problem.
      budget = d * max_weight
    end

    # Return a new instance.
    new(values, weights, m, budget, max_weight)
  end
end

values(i::BudgetedMSetInstance) = i.values
weights(i::BudgetedMSetInstance) = i.weights
m(i::BudgetedMSetInstance) = i.m
budget(i::BudgetedMSetInstance) = i.budget
max_weight(i::BudgetedMSetInstance) = i.max_weight

value(i::BudgetedMSetInstance, o::Int) = values(i)[o]
values(i::BudgetedMSetInstance, o) = values(i)[o]
weight(i::BudgetedMSetInstance, o::Int) = weights(i)[o]
weights(i::BudgetedMSetInstance, o) = weights(i)[o]
dimension(i::BudgetedMSetInstance) = length(values(i))

struct BudgetedMSetSolution
  instance::BudgetedMSetInstance
  items::Vector{Int} # Indices to the chosen items.
  state::Array{Float64, 3} # Data structure built by the dynamic-programming recursion.
  solutions::Dict{Tuple{Int, Int, Int}, Vector{Int}} # From the indices of state to the corresponding solution.
end

function value(s::BudgetedMSetSolution)
  return sum(s.instance.values[i] for i in s.items)
end

function items(s::BudgetedMSetSolution, budget::Int)
  return s.solutions[s.instance.m, 0, budget]
end

function items_all_budgets(s::BudgetedMSetSolution, max_budget::Int)
  sol = Dict{Int, Vector{Int}}()
  m = s.instance.m
  for budget in 0:max_budget
    sol[budget] = s.solutions[m, 0, budget]
  end
  return sol
end

function value(s::BudgetedMSetSolution, budget::Int)
  its = items(s, budget)
  if -1 in its
    return -Inf
  end
  return sum(s.instance.values[i] for i in its)
end
