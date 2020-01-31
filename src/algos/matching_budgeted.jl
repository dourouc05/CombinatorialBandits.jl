struct BudgetedBipartiteMatchingInstance{T, U}
  matching::BipartiteMatchingInstance{T}
  weights::Dict{Edge{T}, U}
  budget::U

  function BudgetedBipartiteMatchingInstance(graph::AbstractGraph{T}, rewards::Dict{Edge{T}, Float64}, weights::Dict{Edge{T}, U}, budget::U) where {T, U}
    matching = BipartiteMatchingInstance(graph, rewards)
    return new{T, U}(matching, weights, budget)
  end
end

function _budgeted_bipartite_matching_compute_value(i::Union{BipartiteMatchingInstance{T}, BudgetedBipartiteMatchingInstance{T, U}}, solution::Vector{Edge{T}}) where T
  return sum(i.matching.rewards[(e in keys(i.rewards)) ? e : reverse(e)] for e in solution)
end

function _budgeted_bipartite_matching_compute_weight(i::BudgetedBipartiteMatchingInstance{T, U}, solution::Vector{Edge{T}}) where {T, U}
  return sum(i.weights[(e in keys(i.weights)) ? e : reverse(e)] for e in solution)
end

abstract type BudgetedBipartiteMatchingSolution{T, U}
  # instance::BudgetedBipartiteMatchingInstance{T, U}
  # solution::Vector{Edge{T}}
  # value::Float64
end

struct BudgetedBipartiteMatchingLagrangianSolution{T, U} <: BudgetedBipartiteMatchingSolution{T, U}
  # Used to store important temporary results from solving the Lagrangian dual.
  instance::BudgetedBipartiteMatchingInstance{T, U}
  solution::Vector{Edge{T}}
  λ::Float64
  value::Float64
  λmax::Float64 # No dual value higher than this is useful (i.e. they all yield the same solution).
end

struct SimpleBudgetedBipartiteMatchingSolution{T, U} <: BudgetedBipartiteMatchingSolution{T, U}
  instance::BudgetedBipartiteMatchingInstance{T, U}
  solution::Vector{Edge{T}}
  value::Float64

  function SimpleBudgetedBipartiteMatchingSolution(instance::BudgetedBipartiteMatchingInstance{T, U}, solution::Vector{Edge{T}}) where {T, U}
    if length(solution) > 0
      return new{T, U}(instance, solution, _budgeted_bipartite_matching_compute_value(instance, solution))
    else
      return new{T, U}(instance, solution, -Inf)
    end
  end

  function SimpleBudgetedBipartiteMatchingSolution(instance::BudgetedBipartiteMatchingInstance{T, U}) where {T, U}
    # No feasible solution.
    return new{T, U}(instance, edgetype(instance.matching.graph)[], -Inf)
  end
end

function matching_hungarian_budgeted_lagrangian(i::BudgetedBipartiteMatchingInstance{T, U}, λ::Float64) where {T, U}
  # Solve the subproblem for one value of the dual multiplier λ:
  #     l(λ) = \max_{x matching} (rewards + λ weights) x - λ budget.
  bmi_rewards = Dict{Edge{T}, Float64}(e => i.matching.rewards[e] + λ * i.weights[e] for e in keys(i.rewards))
  bmi = BipartiteMatchingInstance(i.matching, bmi_rewards)
  bmi_sol = st_prim(bmi)
  return _budgeted_bipartite_matching_compute_value(i, bmi_sol.solution) - λ * i.budget, bmi_sol.solution
end

function matching_hungarian_budgeted_lagrangian_search(i::BudgetedBipartiteMatchingInstance{T, U}, ε::Float64) where {T, U}
  # Approximately solve the problem \min_{l ≥ 0} l(λ), where
  #     l(λ) = \max_{x smatching} (rewards + λ weights) x - λ budget.
  # This problem is the Lagrangian dual of the budgeted maximum spanning-tree problem:
  #     \max_{x matching} rewards x  s.t.  weights x >= budget.
  # This algorithm provides no guarantee on the optimality of the solution.

  # TODO: generalise this function?

  # Initial set of values for λ. The optimum is guaranteed to be contained in this interval.
  weights_norm_inf = maximum(values(i.weights)) # Maximum weight.
  m = min(i.matching.n_left, i.matching.n_right) # Maximum number of items in a solution. Easy to compute for a matching!
  λmax = Float64(weights_norm_inf * m + 1)

  λlow = 0.0
  λhigh = λmax

  # Perform a golden-ratio search.
  while (λhigh - λlow) > ε
    λmidlow = λhigh - (λhigh - λlow) / MathConstants.φ
    λmidhigh = λlow + (λhigh - λlow) / MathConstants.φ
    vmidlow, _ = matching_hungarian_budgeted_lagrangian(i, λmidlow)
    vmidhigh, _ = matching_hungarian_budgeted_lagrangian(i, λmidhigh)

    if vmidlow < vmidhigh
      λhigh = λmidhigh
      vhigh = vmidhigh
    else
      λlow = λmidlow
      vlow = vmidlow
    end
  end

  vlow, bmlow = matching_hungarian_budgeted_lagrangian(i, λlow)
  vhigh, bmhigh = matching_hungarian_budgeted_lagrangian(i, λhigh)
  if vlow > vhigh
    return BudgetedBipartiteMatchingLagrangianSolution(i, bmlow, λlow, vlow, λmax)
  else
    return BudgetedBipartiteMatchingLagrangianSolution(i, bmhigh, λhigh, vhigh, λmax)
  end
end
