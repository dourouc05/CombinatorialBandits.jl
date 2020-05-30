solve(i::BudgetedBipartiteMatchingInstance{T}, ::LagrangianAlgorithm, ε; kwargs...) where T = matching_hungarian_budgeted_lagrangian_search(i, ε; kwargs...)
solve(i::BudgetedBipartiteMatchingInstance{T}, ::LagrangianRefinementAlgorithm; kwargs...) where T = matching_hungarian_budgeted_lagrangian_refinement(i; kwargs...)
solve(i::BudgetedBipartiteMatchingInstance{T}, ::IteratedLagrangianRefinementAlgorithm; kwargs...) where T = matching_hungarian_budgeted_lagrangian_approx_half(i; kwargs...)

approximation_term(::BudgetedBipartiteMatchingInstance{T}, ::LagrangianAlgorithm) where T = NaN
approximation_ratio(::BudgetedBipartiteMatchingInstance{T}, ::LagrangianAlgorithm) where T = NaN

approximation_term(i::BudgetedBipartiteMatchingInstance{T}, ::LagrangianRefinementAlgorithm) where T = maximum(values(i.matching.rewards))
approximation_ratio(::BudgetedBipartiteMatchingInstance{T}, ::LagrangianRefinementAlgorithm) where T = NaN

approximation_term(::BudgetedBipartiteMatchingInstance{T}, ::IteratedLagrangianRefinementAlgorithm) where T = NaN
approximation_ratio(::BudgetedBipartiteMatchingInstance{T}, ::IteratedLagrangianRefinementAlgorithm) where T = 0.5

struct BudgetedBipartiteMatchingLagrangianSolution{T, U} <: BudgetedBipartiteMatchingSolution{T, U}
  # Used to store important temporary results from solving the Lagrangian dual.
  instance::BudgetedBipartiteMatchingInstance{T, U}
  solution::Vector{Edge{T}}
  λ::Float64 # Optimum dual multiplier.
  value::Float64 # Optimum value of the dual problem (i.e. with penalised constraint).
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
  bmi_rewards = Dict{Edge{T}, Float64}(e => i.matching.reward[e] + λ * i.weight[e] for e in keys(i.matching.reward))
  bmi = BipartiteMatchingInstance(i.matching, bmi_rewards)
  bmi_sol = matching_hungarian(bmi)
  return _budgeted_bipartite_matching_compute_value(bmi, bmi_sol.solution) - λ * i.budget, bmi_sol.solution
end

function matching_hungarian_budgeted_lagrangian_search(i::BudgetedBipartiteMatchingInstance{T, U}, ε::Float64) where {T, U}
  # Approximately solve the problem \min_{l ≥ 0} l(λ), where
  #     l(λ) = \max_{x smatching} (rewards + λ weights) x - λ budget.
  # This problem is the Lagrangian dual of the budgeted maximum spanning-tree problem:
  #     \max_{x matching} rewards x  s.t.  weights x >= budget.
  # This algorithm provides no guarantee on the optimality of the solution.

  # TODO: generalise this function? Highly similar to the one for spanning trees.

  # Initial set of values for λ. The optimum is guaranteed to be contained in this interval.
  weights_norm_inf = maximum(values(i.weight)) # Maximum weight.
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
  if vlow < vhigh
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
  feasible_rewards = Dict{Edge{T}, Float64}(e => i.weight[e] for e in keys(i.weight))
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
      infeasible_rewards = Dict{Edge{T}, Float64}(e => - i.weight[e] for e in keys(i.weight))
      infeasible_solution = matching_hungarian(BipartiteMatchingInstance(i.matching.graph, infeasible_rewards)).solution

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

  already_seen_solutions = Set{Set{Edge{T}}}()
  push!(already_seen_solutions, Set(x⁺))
  push!(already_seen_solutions, Set(x⁻))

  # Iterative refinement. Stop as soon as there is a difference of at most one edge between the two solutions.
  while _solution_symmetric_difference_size(x⁺, x⁻) > 4
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

    while true
      # Exit conditions:
      # - If the XOR is a path: consume all edges; at some point, will no more
      #   be possible to find the next one
      # - Otherwise, once a node is met twice, break.

      # If the current edge is in the solution, remove it. Otherwise, add it.
      if current_edge in new_x
        filter!(e -> e != current_edge, new_x)
      else
        push!(new_x, current_edge)
      end

      # Go to the next edge, following a cycle or a path.
      if current_vertex == src(current_edge)
        current_vertex = dst(current_edge)
      else
        current_vertex = src(current_edge)
      end
      current_edge_idx = findfirst(e -> src(e) == current_vertex || dst(e) == current_vertex, xor)
      if isnothing(current_edge_idx)
        break
      end
      current_edge = xor[current_edge_idx]
      deleteat!(xor, current_edge_idx)

      # Check for a cycle.
      if current_vertex in all_vertices
        break
      end
      push!(all_vertices, current_vertex)
    end

    # If this procedure cannot propose a new solution, it has stalled and will not recover.
    if Set(new_x) in already_seen_solutions
      break
    else
      push!(already_seen_solutions, Set(new_x))
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

function _edge_any_end_match(e::Edge{T}, t::Edge{T}) where T
  return src(e) == src(t) || dst(e) == dst(t) || src(e) == dst(t) || dst(e) == src(t)
end

function matching_hungarian_budgeted_lagrangian_approx_half(i::BudgetedBipartiteMatchingInstance{T, U}; kwargs...) where {T, U}
  # Approximately solve the following problem:
  #     \max_{x matching} rewards x  s.t.  weights x >= budget
  # This algorithm provides a multiplicative approximation to this problem. If x* is the optimum solution and x~ the one
  # returned by this algorithm,
  #     weights x* >= budget   and   weights x~ >= budget                 (the returned solution is feasible)
  #     rewards x~ >= (1 - ε) rewards x*                                  (multiplicative approximation)
  # The parameter ε is not tuneable, but rather fixed to 1/2.

  # This is based on http://people.idsia.ch/~grandoni/Pubblicazioni/BBGS08ipco.pdf, Theorem 1.

  # If there are too few vertices, not much to do. The smallest side of the bipartite graph must have at least 4 vertices,
  # so that 4 of them can be fixed.
  if i.matching.n_left <= 4 || i.matching.n_right <= 4
    # TODO: Should rather perform an exhaustive exploration.
    return matching_hungarian_budgeted_lagrangian_refinement(i; kwargs...)
  end

  # For each combination of four distinct edges, force these four edges to be part of the solution and discard all edges with a higher value.
  # If selecting twice the same edge: no need to go further.
  # If any end of two selected edges are the same: this cannot lead to a feasible solution.
  # Both tests are implemented in _edge_any_end_match.
  best_sol = nothing
  for e1 in edges(i.matching.graph)
    for e2 in edges(i.matching.graph)
      if _edge_any_end_match(e1, e2)
        continue
      end

      for e3 in edges(i.matching.graph)
        if _edge_any_end_match(e1, e3) || _edge_any_end_match(e2, e3)
          continue
        end

        for e4 in edges(i.matching.graph)
          if _edge_any_end_match(e1, e4) || _edge_any_end_match(e2, e4) || _edge_any_end_match(e3, e4)
            continue
          end

          # Filter out the edges that have a higher value than any of these two edges. Give a very large reward to them both.
          cutoff = min(i.matching.reward[e1], i.matching.reward[e2], i.matching.reward[e3], i.matching.reward[e4])
          reward = copy(i.matching.reward)
          filter!(kv -> kv[2] <= cutoff, reward)
          filter!(kv -> ! _edge_any_end_match(kv[1], e1), reward)
          filter!(kv -> ! _edge_any_end_match(kv[1], e2), reward)
          filter!(kv -> ! _edge_any_end_match(kv[1], e3), reward)
          filter!(kv -> ! _edge_any_end_match(kv[1], e4), reward)
          reward[e1] = reward[e2] = reward[e3] = reward[e4] = prevfloat(Inf)

          if length(keys(reward)) == 4 # Nothing left? Skip.
            continue
          end

          graph = SimpleGraph(nv(i.matching.graph))
          for e in keys(reward)
            add_edge!(graph, e)
          end

          weight = Dict(e => i.weight[e] for e in keys(reward))

          # Solve this subproblem.
          bbmi = BudgetedBipartiteMatchingInstance(graph, reward, weight, i.budget)
          sol = matching_hungarian_budgeted_lagrangian_refinement(bbmi; kwargs...)

          # This subproblem is infeasible. Maybe it's because the overall problem is infeasible or just because too many
          # edges were removed.
          if length(sol.solution) == 0
            continue
          end

          # Impossible to have a feasible solution with these two edges, probably because of the budget constraint.
          # It's very unlikely that this happens, due to the checks already performed when looping.
          if ! (e1 in sol.solution) || ! (e2 in sol.solution) || ! (e3 in sol.solution) || ! (e4 in sol.solution)
            continue
          end

          # Only keep the best solution.
          if best_sol == nothing || _budgeted_bipartite_matching_compute_value(i, sol.solution) > _budgeted_bipartite_matching_compute_value(i, best_sol.solution)
            # sol's instance is the one used internally for the subproblems.
            best_sol = SimpleBudgetedBipartiteMatchingSolution(i, sol.solution)
          end
        end
      end
    end
  end

  if best_sol !== nothing
    return best_sol
  else
    return SimpleBudgetedBipartiteMatchingSolution(i)
  end
end