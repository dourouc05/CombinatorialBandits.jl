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
  bmi_sol = matching_hungarian(bmi)
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

function matching_hungarian_budgeted_lagrangian_refinement(i::BudgetedBipartiteMatchingInstance{T, U};
                                                           ε::Float64=1.0e-3, ζ⁻::Float64=0.2, ζ⁺::Float64=5.0,
                                                           stalling⁻::Float64=1.0e-5) where {T, U}
  # Approximately solve the following problem:
  #     \max_{x matching} rewards x  s.t.  weights x >= budget
  # This algorithm provides an additive approximation to this problem. If x* is the optimum solution and x~ the one
  # returned by this algorithm,
  #     weights x* >= budget   and   weights x~ >= budget                 (the returned solution is feasible)
  #     rewards x~ >= rewards x* - 2 \max{e edge} reward[e]               (additive approximation)
  # This is based on http://people.idsia.ch/~grandoni/Pubblicazioni/BBGS08ipco.pdf, Lemma 4.

  # Check assumptions.
  if ζ⁻ >= 1.0
    error("ζ⁻ must be strictly less than 1.0: the dual multiplier λ is multiplied by ζ⁻ to reach an infeasible solution by less penalising the budget constraint.")
  end
  if ζ⁺ <= 1.0
    error("ζ⁺ must be strictly greater than 1.0: the dual multiplier λ is multiplied by ζ⁺ to reach a feasible solution by penalising more the budget constraint.")
  end

  # Ensure the problem is feasible by only considering the budget constraint.
  feasible_rewards = Dict{Edge{T}, Float64}(e => i.weights[e] for e in keys(i.rewards))
  feasible_instance = BipartiteMatchingInstance(i.matching, feasible_rewards)
  feasible_solution = matching_hungarian(feasible_instance)
  if _budgeted_bipartite_matching_compute_value(feasible_instance, feasible_solution.solution) < i.budget
    # By maximising the left-hand side of the budget constraint, impossible to reach the target budget. No solution!
    return SimpleBudgetedBipartiteMatchingSolution(i)
  end

  # Solve the Lagrangian relaxation to optimality.
  lagrangian = matching_hungarian_budgeted_lagrangian_search(i, ε)
  λ0, v0, bm0 = lagrangian.λ, lagrangian.value, lagrangian.solution
  λmax = lagrangian.λmax
  b0 = _budgeted_bipartite_matching_compute_weight(i, bm0) # Budget consumption of this first solution.

  # If already respecting the budget constraint exactly, done!
  if b0 == i.budget
    return SimpleBudgetedBipartiteMatchingSolution(i, bm0)
  end

  # Find two solutions: one above the budget x⁺ (i.e. respecting the constraint), the other not x⁻.
  x⁺, x⁻ = nothing, nothing

  λi = λ0
  if b0 > i.budget
    x⁺ = bm0
    @assert _budgeted_bipartite_matching_compute_weight(i, x⁺) > i.budget

    stalling = false
    while true
      # Penalise less the constraint: it should no more be satisfied.
      λi *= ζ⁻
      _, sti = matching_hungarian_budgeted_lagrangian(i, λi)
      if _budgeted_bipartite_matching_compute_weight(i, sti) < i.budget
        x⁻ = sti
        break
      end

      # Is the process stalling?
      if λi <= stalling⁻
        stalling = true
        break
      end
    end

    # Specific handling of stallings.
    if stalling # First test: don't penalise the constraint at all.
      _, sti = matching_hungarian_budgeted_lagrangian(i, 0.0)
      new_budget = _budgeted_bipartite_matching_compute_weight(i, sti)
      if new_budget < i.budget
        x⁻ = sti
        stalling = false
      end
    end

    if stalling # Second test: minimise the left-hand side of the budget constraint, in hope of finding a feasible solution.
      # This process is highly similar to the computation of feasible_solution, but with a reverse objective function.
      infeasible_rewards = Dict{Edge{T}, Float64}(e => - i.weights[e] for e in keys(i.weights))
      infeasible_solution = matching_hungarian(BipartiteMatchingInstance(i.graph, infeasible_rewards)).solution

      if _budgeted_bipartite_matching_compute_weight(i, infeasible_solution) < i.budget
        x⁻ = infeasible_solution
        stalling = false
      end
    end

    if stalling # Third: decide there is no solution strictly below the budget. No refinement is possible.
      # As x⁺ is feasible, return it.
      return SimpleBudgetedBipartiteMatchingSolution(i, x⁺)
    end
  else
    x⁻ = bm0
    @assert _budgeted_bipartite_matching_compute_weight(i, x⁻) < i.budget

    while true
      # Penalise more the constraint: it should become satisfied at some point.
      λi *= ζ⁺
      _, sti = matching_hungarian_budgeted_lagrangian(i, λi)
      if _budgeted_bipartite_matching_compute_weight(i, sti) >= i.budget
        x⁺ = sti
        break
      end

      # Is the process stalling? If so, reuse feasible_solution, which is guaranteed to be feasible.
      if λi >= λmax
        x⁺ = feasible_solution.solution
        break
      end
    end
  end

  # Normalise the solutions: the edges point from the left part to the right part of the bipartite graph.
  sort_edge(e::Edge{T}) where T = (i.matching.partition[src(e)] == 1) ? e : reverse(e)
  x⁺ = [sort_edge(e) for e in x⁺]
  x⁻ = [sort_edge(e) for e in x⁻]

  # Iterative refinement. Stop as soon as there is a difference of at most one edge between the two solutions.
  while _solution_symmetric_difference_size(x⁺, x⁻) > 2
    # Enforce the loop invariant.
    @assert x⁺ !== nothing
    @assert x⁻ !== nothing
    @assert _budgeted_bipartite_matching_compute_weight(i, x⁺) >= i.budget # Feasible.
    @assert _budgeted_bipartite_matching_compute_weight(i, x⁻) < i.budget # Infeasible.

    # Switch elements from one solution to another: use the XOR between the two solutions as an augmenting path/cycle.
    only_in_x⁺, only_in_x⁻ = _solution_symmetric_difference(x⁺, x⁻)
    xor = vcat(only_in_x⁺, only_in_x⁻)

    current_edge = first(xor)
    current_vertex = src(current_edge)
    all_vertices = Set{T}(current_vertex)
    xor = xor[2:end]

    new_x = copy(x⁺)
    while length(xor) > 0 # If the XOR is a path: consume all edges; otherwise, once a node is met twice, break.
      # If the current edge is in the solution, remove it. Otherwise, add it.
      if current_edge in new_x
        filter!(e -> e != current_edge, new_x)
      else
        push!(new_x, current_edge)
      end

      # Go to the next edge, following a cycle or a path.
      current_vertex = dst(current_edge)
      current_edge_idx = findfirst(e -> src(e) == current_vertex, xor)
      current_edge = xor[current_edge]
      xor = vcat(xor[1:(current_edge_idx - 1)], xor[(current_edge + 1):end])

      # Check for a cycle.
      if current_vertex in all_vertices
        break
      end
      push!(all_vertices, current_vertex)
    end

    # Replace one of the two solutions, depending on whether this solution is feasible (x⁺) or not (x⁻).
    if _budgeted_bipartite_matching_compute_weight(i, new_x) >= i.budget
      x⁺ = new_x
    else
      x⁻ = new_x
    end
  end

  # Done!
  return SimpleBudgetedBipartiteMatchingSolution(i, x⁺)
end

# This is based on http://people.idsia.ch/~grandoni/Pubblicazioni/BBGS08ipco.pdf, Theorem 1.
