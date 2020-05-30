solve(i::BudgetedSpanningTreeInstance{T}, ::LagrangianAlgorithm, ε; kwargs...) where T = st_prim_budgeted_lagrangian_search(i, ε; kwargs...)
solve(i::BudgetedSpanningTreeInstance{T}, ::LagrangianRefinementAlgorithm; kwargs...) where T = st_prim_budgeted_lagrangian_refinement(i; kwargs...)
solve(i::BudgetedSpanningTreeInstance{T}, ::IteratedLagrangianRefinementAlgorithm; kwargs...) where T = st_prim_budgeted_lagrangian_approx_half(i; kwargs...)

approximation_term(::BudgetedSpanningTreeInstance{T}, ::LagrangianAlgorithm) where T = NaN
approximation_ratio(::BudgetedSpanningTreeInstance{T}, ::LagrangianAlgorithm) where T = NaN

approximation_term(i::BudgetedSpanningTreeInstance{T}, ::LagrangianRefinementAlgorithm) where T = maximum(values(i.rewards))
approximation_ratio(::BudgetedSpanningTreeInstance{T}, ::LagrangianRefinementAlgorithm) where T = NaN

approximation_term(::BudgetedSpanningTreeInstance{T}, ::IteratedLagrangianRefinementAlgorithm) where T = NaN
approximation_ratio(::BudgetedSpanningTreeInstance{T}, ::IteratedLagrangianRefinementAlgorithm) where T = 0.5

function _budgeted_spanning_tree_compute_weight(i::BudgetedSpanningTreeInstance{T, U}, tree::Vector{Edge{T}}) where {T, U}
  return sum(i.weights[(e in keys(i.weights)) ? e : reverse(e)] for e in tree)
end

struct BudgetedSpanningTreeLagrangianSolution{T, U} <: BudgetedSpanningTreeSolution{T, U}
  # Used to store important temporary results from solving the Lagrangian dual.
  instance::BudgetedSpanningTreeInstance{T, U}
  tree::Vector{Edge{T}}
  λ::Float64 # Optimum dual multiplier.
  value::Float64 # Optimum value of the dual problem (i.e. with penalised constraint).
  λmax::Float64 # No dual value higher than this is useful (i.e. they all yield the same solution).
end

function st_prim_budgeted_lagrangian(i::BudgetedSpanningTreeInstance{T, U}, λ::Float64) where {T, U}
  # Solve the subproblem for one value of the dual multiplier λ:
  #     l(λ) = \max_{x spanning tree} (rewards + λ weights) x - λ budget.
  sti_rewards = Dict{Edge{T}, Float64}(e => i.rewards[e] + λ * i.weights[e] for e in keys(i.rewards))
  sti = SpanningTreeInstance(i.graph, sti_rewards)
  sti_sol = st_prim(sti)
  return _budgeted_spanning_tree_compute_value(sti, sti_sol.tree) - λ * i.budget, sti_sol.tree
end

function st_prim_budgeted_lagrangian_search(i::BudgetedSpanningTreeInstance{T, U}, ε::Float64) where {T, U}
  # Approximately solve the problem \min_{l ≥ 0} l(λ), where
  #     l(λ) = \max_{x spanning tree} (rewards + λ weights) x - λ budget.
  # This problem is the Lagrangian dual of the budgeted maximum spanning-tree problem:
  #     \max_{x spanning tree} rewards x  s.t.  weights x >= budget.
  # This algorithm provides no guarantee on the optimality of the solution.

  # Initial set of values for λ. The optimum is guaranteed to be contained in this interval.
  weights_norm_inf = maximum(values(i.weights)) # Maximum weight.
  m = nv(i.graph) - 1 # Maximum number of items in a solution. Easy to compute for a spanning tree!
  λmax = Float64(weights_norm_inf * m + 1)

  λlow = 0.0
  λhigh = λmax

  # Perform a golden-ratio search.
  while (λhigh - λlow) > ε
    λmidlow = λhigh - (λhigh - λlow) / MathConstants.φ
    λmidhigh = λlow + (λhigh - λlow) / MathConstants.φ
    vmidlow, _ = st_prim_budgeted_lagrangian(i, λmidlow)
    vmidhigh, _ = st_prim_budgeted_lagrangian(i, λmidhigh)

    if vmidlow < vmidhigh
      λhigh = λmidhigh
      vhigh = vmidhigh
    else
      λlow = λmidlow
      vlow = vmidlow
    end
  end

  vlow, stlow = st_prim_budgeted_lagrangian(i, λlow)
  vhigh, sthigh = st_prim_budgeted_lagrangian(i, λhigh)
  if vlow < vhigh
    return BudgetedSpanningTreeLagrangianSolution(i, stlow, λlow, vlow, λmax)
  else
    return BudgetedSpanningTreeLagrangianSolution(i, sthigh, λhigh, vhigh, λmax)
  end
end

function st_prim_budgeted_lagrangian_refinement(i::BudgetedSpanningTreeInstance{T, U};
                                                ε::Float64=1.0e-3, ζ⁻::Float64=0.2, ζ⁺::Float64=5.0,
                                                stalling⁻::Float64=1.0e-5) where {T, U}
  # Approximately solve the following problem:
  #     \max_{x spanning tree} rewards x  s.t.  weights x >= budget
  # This algorithm provides an additive approximation to this problem. If x* is the optimum solution and x~ the one
  # returned by this algorithm,
  #     weights x* >= budget   and   weights x~ >= budget                 (the returned solution is feasible)
  #     rewards x~ >= rewards x* - \max{e edge} reward[e]                 (additive approximation)

  # Check assumptions.
  if ζ⁻ >= 1.0
    error("ζ⁻ must be strictly less than 1.0: the dual multiplier λ is multiplied by ζ⁻ to reach an infeasible solution by less penalising the budget constraint.")
  end
  if ζ⁺ <= 1.0
    error("ζ⁺ must be strictly greater than 1.0: the dual multiplier λ is multiplied by ζ⁺ to reach a feasible solution by penalising more the budget constraint.")
  end

  # Ensure the problem is feasible by only considering the budget constraint.
  feasible_rewards = Dict{Edge{T}, Float64}(e => i.weights[e] for e in keys(i.rewards))
  feasible_instance = SpanningTreeInstance(i.graph, feasible_rewards)
  feasible_solution = st_prim(feasible_instance)
  if _budgeted_spanning_tree_compute_value(feasible_instance, feasible_solution.tree) < i.budget
    # By maximising the left-hand side of the budget constraint, impossible to reach the target budget. No solution!
    return SimpleBudgetedSpanningTreeSolution(i)
  end

  # Solve the Lagrangian relaxation to optimality.
  lagrangian = st_prim_budgeted_lagrangian_search(i, ε)
  λ0, v0, st0 = lagrangian.λ, lagrangian.value, lagrangian.tree
  λmax = lagrangian.λmax
  b0 = _budgeted_spanning_tree_compute_weight(i, st0) # Budget consumption of this first solution.

  # If already respecting the budget constraint exactly, done!
  if b0 == i.budget
    return SimpleBudgetedSpanningTreeSolution(i, st0)
  end

  # Find two solutions: one above the budget x⁺ (i.e. respecting the constraint), the other not x⁻.
  x⁺, x⁻ = nothing, nothing

  λi = λ0
  if b0 > i.budget
    x⁺ = st0
    @assert _budgeted_spanning_tree_compute_weight(i, x⁺) > i.budget

    stalling = false
    while true
      # Penalise less the constraint: it should no more be satisfied.
      λi *= ζ⁻
      _, sti = st_prim_budgeted_lagrangian(i, λi)
      if _budgeted_spanning_tree_compute_weight(i, sti) < i.budget
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
      _, sti = st_prim_budgeted_lagrangian(i, 0.0)
      new_budget = _budgeted_spanning_tree_compute_weight(i, sti)
      if new_budget < i.budget
        x⁻ = sti
        stalling = false
      end
    end

    if stalling # Second test: minimise the left-hand side of the budget constraint, in hope of finding a feasible solution.
      # This process is highly similar to the computation of feasible_solution, but with a reverse objective function.
      infeasible_rewards = Dict{Edge{T}, Float64}(e => - i.weights[e] for e in keys(i.weights))
      infeasible_solution = st_prim(SpanningTreeInstance(i.graph, infeasible_rewards)).tree

      if _budgeted_spanning_tree_compute_weight(i, infeasible_solution) < i.budget
        x⁻ = infeasible_solution
        stalling = false
      end
    end

    if stalling # Third: decide there is no solution strictly below the budget. No refinement is possible.
      # As x⁺ is feasible, return it.
      return SimpleBudgetedSpanningTreeSolution(i, x⁺)
    end
  else
    x⁻ = st0
    @assert _budgeted_spanning_tree_compute_weight(i, x⁻) < i.budget

    while true
      # Penalise more the constraint: it should become satisfied at some point.
      λi *= ζ⁺
      _, sti = st_prim_budgeted_lagrangian(i, λi)
      if _budgeted_spanning_tree_compute_weight(i, sti) >= i.budget
        x⁺ = sti
        break
      end

      # Is the process stalling? If so, reuse feasible_solution, which is guaranteed to be feasible.
      if λi >= λmax
        x⁺ = feasible_solution.tree
        break
      end
    end
  end

  # Normalise the solutions: the input graph is undirected, the direction of the edges is not important.
  # In case one solution has the edge v -> w and the other one w -> v, make them equal. s
  sort_edge(e::Edge{T}) where T = (src(e) < dst(e)) ? e : reverse(e)
  x⁺ = [sort_edge(e) for e in x⁺]
  x⁻ = [sort_edge(e) for e in x⁻]

  # Iterative refinement. Stop as soon as there is a difference of at most one edge between the two solutions.
  while _solution_symmetric_difference_size(x⁺, x⁻) > 2
    # Enforce the loop invariant.
    @assert x⁺ !== nothing
    @assert x⁻ !== nothing
    @assert _budgeted_spanning_tree_compute_weight(i, x⁺) >= i.budget # Feasible.
    @assert _budgeted_spanning_tree_compute_weight(i, x⁻) < i.budget # Infeasible.

    # Switch elements from one solution to another.
    only_in_x⁺, only_in_x⁻ = _solution_symmetric_difference(x⁺, x⁻)
    e1 = first(only_in_x⁺)
    e2 = first(only_in_x⁻)

    # Create the new solution (don't erase x⁺ nor x⁻: only one of the two will be forgotten, the other will be kept).
    new_x = copy(x⁺)
    filter!(e -> e != e1, new_x)
    push!(new_x, e2)

    # Replace one of the two solutions, depending on whether this solution is feasible (x⁺) or not (x⁻).
    if _budgeted_spanning_tree_compute_weight(i, new_x) >= i.budget
      x⁺ = new_x
    else
      x⁻ = new_x
    end
  end

  # Done!
  return SimpleBudgetedSpanningTreeSolution(i, x⁺)
end

function st_prim_budgeted_lagrangian_approx_half(i::BudgetedSpanningTreeInstance{T, U}; kwargs...) where {T, U}
  # Approximately solve the following problem:
  #     \max_{x spanning tree} rewards x  s.t.  weights x >= budget
  # This algorithm provides a multiplicative approximation to this problem. If x* is the optimum solution and x~ the one
  # returned by this algorithm,
  #     weights x* >= budget   and   weights x~ >= budget                 (the returned solution is feasible)
  #     rewards x~ >= rewards x* / 2                                      (multiplicative approximation)

  # Based on the same ideas as https://doi.org/10.1007/s10107-009-0307-4

  # If there are too few edges, not much to do.
  if ne(i.graph) <= 2
    return st_prim_budgeted_lagrangian_refinement(i; kwargs...)
  end

  # For each pair of edges, force these two edges to be part of the solution and discard all edges with a higher value.
  best_sol = nothing
  for e1 in edges(i.graph)
    for e2 in edges(i.graph)
      # (e1, e2) must be a pair of distinct edges.
      if (src(e1) == src(e2) && dst(e1) == dst(e2)) || (src(e1) == dst(e2) && src(e1) == dst(e2))
        continue
      end

      # Filter out the edges that have a higher value than any of these two edges. Give a very large reward to them both.
      cutoff = min(i.rewards[e1], i.rewards[e2])
      rewards = filter(kv -> kv[2] < cutoff, i.rewards)
      rewards[e1] = rewards[e2] = prevfloat(Inf)

      graph = SimpleGraph(nv(i.graph))
      for e in keys(rewards)
        add_edge!(graph, e)
      end

      weights = Dict(e => i.weights[e] for e in keys(rewards))

      # No other simplification can be made, unfortunately: all other edges might participate in the optimum solution.
      # (Only exception: e1 and e2 are incident to the same vertex; any edge linking the other extremities of these edges
      # can be removed, but that's only one edge at most.)
      # As e1 and e2 have the best reward (as they are really bumped), they must be in any optimum solution.

      # Solve this subproblem.
      bsti = BudgetedSpanningTreeInstance(graph, rewards, weights, i.budget)
      sol = st_prim_budgeted_lagrangian_refinement(bsti; kwargs...)

      # This subproblem is infeasible. Maybe it's because the overall problem is infeasible or just because too many
      # edges were removed.
      if length(sol.tree) == 0
        continue
      end

      # Impossible to have a feasible solution with these two edges, probably because of the budget constraint.
      # Due to the direction of the budget constraint (>= budget), it is not possible to check for feasibility
      # before solving an instance.
      if ! (e1 in sol.tree) || ! (e2 in sol.tree)
        continue
      end

      # Only keep the best solution.
      if best_sol === nothing || sol.value > best_sol.value
        # sol's instance is the one used internally for the subproblems.
        best_sol = SimpleBudgetedSpanningTreeSolution(i, sol.tree)
      end
    end
  end

  if best_sol !== nothing
    return best_sol
  else
    return SimpleBudgetedSpanningTreeSolution(i)
  end
end