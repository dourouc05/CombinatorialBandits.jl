solve(i::BudgetedElementaryPathInstance{T}, ::DynamicProgramming; kwargs...) where T = budgeted_lp_dp(i; kwargs...)

function budgeted_lp_dp(i::BudgetedElementaryPathInstance{T}) where T
  V = Dict{Tuple{T, Int}, Float64}()
  S = Dict{Tuple{T, Int}, Vector{Edge{T}}}()

  # Initialise. For β = 0, this is exactly Bellman-Ford algorithm with costs
  # (instead of rewards). Otherwise, use the same initialisation as
  # Bellman-Ford.
  β0 = lp_dp(ElementaryPathInstance(graph(i), rewards(i), src(i), dst(i)))
  for v in vertices(graph(i))
    S[v, 0] = β0.solutions[v]
    V[v, 0] = length(S[v, 0]) == 0 ? 0 : sum(rewards(i)[e] for e in S[v, 0])
  end

  for β in 1:budget(i)
    for v in vertices(graph(i))
      V[v, β] = -Inf
      S[v, β] = Edge{T}[]
    end

    V[src(i), β] = 0.0
  end

  # Dynamic part.
  for β in 1:budget(i)
    # Loop needed when at least a weight is equal to zero. TODO: remove it when all weights are nonzero?
    while true
      changed = false

      for v in vertices(graph(i))
        for w in inneighbors(graph(i), v)
          # Compute the remaining part of the budget still to use.
          remaining_budget = max(0, β - weight(i, w, v))

          # If the explored subproblem has no solution, skip it.
          if V[w, remaining_budget] == -Inf
            continue
          end

          # If using the solution to the currently explored subproblem would
          # lead to a cycle, skip it.
          if any(src(e) == v for e in S[w, remaining_budget])
            continue
          end

          # Compute the amount of budget already used by the solution to the currently explored subproblem.
          used_budget = 0
          if length(S[w, remaining_budget]) > 0
            used_budget = sum(weight(i, src(e), dst(e)) for e in S[w, remaining_budget])
          end
          if used_budget < remaining_budget
            continue
          end

          # Compute the maximum: is passing through w advantageous?
          if V[w, remaining_budget] + reward(i, w, v) > V[v, β] && used_budget >= remaining_budget
            changed = true

            V[v, β] = V[w, remaining_budget] + reward(i, w, v)
            S[v, β] = vcat(S[w, remaining_budget], Edge(w, v))
          end
        end
      end

      if ! changed
        break
      end
    end
  end

  return BudgetedElementaryPathSolution(i, S[dst(i), budget(i)], V, S)
end