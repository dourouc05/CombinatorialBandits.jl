solve(i::BudgetedMSetInstance, ::DynamicProgramming; kwargs...) = budgeted_msets_dp(i; kwargs...)

function budgeted_msets_dp(i::BudgetedMSetInstance)
  # Consider items above δ.
  # Recurrence relation:
  #  V[m, t, δ] = max { V[m, t, δ + 1],  v_δ + V[m - 1, t - w_δ, δ + 1] }
  #                     \_ no item δ _/ \________ take item δ ________/
  V = Array{Float64, 3}(undef, m(i), dimension(i) + 1, budget(i) + 1)
  S = Dict{Tuple{Int, Int, Int}, Vector{Int}}()

  # Initialise: µ == 1, just take the best element among [δ+1, d], as long as
  # the budget constraint is satisfied; δ == d, take nothing, whatever the
  # budget; β == 0, a simple m-set problem.
  for µ in 1:m(i) # β == 0
    for δ in (dimension(i) - 1):-1:0 # δ < d
      # Due to δ, there may be too few items (with respect to µ).
      # i.e., there are d - δ items to consider.
      m_ = min(dimension(i) - δ, µ)

      items = collect(partialsortperm(values(i, (δ + 1):dimension(i)), 1:m_, rev=true)) .+ δ
      V[µ, δ + 1, 0 + 1] = sum(value(i, o) for o in items)
      S[µ, δ, 0] = items
    end
  end

  for β in 0:budget(i)
    # δ == d
    for µ in 1:m(i)
      if β == 0
        # Optimum solution: take no object.
        V[µ, dimension(i) + 1, β + 1] = 0.0
        S[µ, dimension(i), β] = Int[]
      else
        # No solution.
        V[µ, dimension(i) + 1, β + 1] = -Inf
        S[µ, dimension(i), β] = [-1] # Nonsensical solution.
      end
    end

    # δ < d
    for δ in (dimension(i) - 1):-1:0
      # Filter the available objects (i.e. above δ+1 and satisfying the
      # budget constraint) and retrieve just their indices.
      all_objects = collect(enumerate(weights(i)))
      all_objects = collect(filter(o -> o[1] >= δ + 1, all_objects))
      all_objects = collect(filter(o -> o[2] >= β, all_objects))
      all_objects = collect(t[1] for t in all_objects)

      # No feasible solution?
      if length(all_objects) == 0
        V[1, δ + 1, β + 1] = -Inf
        S[1, δ, β] = [-1] # Nonsensical solution.
        continue
      end

      # Take the best one available object.
      v, x = findmax(collect(value(i, t) for t in all_objects))
      x = all_objects[x]

      V[1, δ + 1, β + 1] = v
      S[1, δ, β] = [x]
    end
  end

  # Dynamic part.
  for β in 1:budget(i)
    for µ in 2:m(i)
      for δ in (dimension(i) - 1):-1:0
        remaining_budget_with_δ = max(0, β - weights(i)[δ + 1])
        take_δ = value(i, δ + 1) + V[µ - 1, δ + 1 + 1, remaining_budget_with_δ + 1]
        dont_take_δ = V[µ, δ + 1 + 1, β + 1]

        both_subproblems_infeasible = -Inf == take_δ && -Inf == dont_take_δ

        if both_subproblems_infeasible
          V[µ, δ + 1, β + 1] = -Inf
          S[µ, δ, β] = [-1]
        elseif take_δ >= dont_take_δ
          V[µ, δ + 1, β + 1] = take_δ
          S[µ, δ, β] = vcat(δ + 1, S[µ - 1, δ + 1, remaining_budget_with_δ])
        else
          V[µ, δ + 1, β + 1] = dont_take_δ
          S[µ, δ, β] = S[µ, δ + 1, β]
        end
      end
    end
  end

  return BudgetedMSetSolution(i, S[m(i), 0, budget(i)], V, S)
end