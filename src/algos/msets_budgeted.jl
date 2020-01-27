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

function _budgeted_msets_lp_sub(i::BudgetedMSetInstance, solver)
  model = Model(solver)
  @variable(model, x[1:length(i.values)], Bin)
  @objective(model, Max, dot(x, values(i)))
  @constraint(model, sum(x) <= m(i))
  @constraint(model, c, dot(x, weights(i)) >= 0)

  set_silent(model)

  return model, x, c
end

function budgeted_msets_lp(i::BudgetedMSetInstance; solver=nothing, β::Int=budget(i))
  # Solve for all budgets at once, even though it is not really more efficient than looping outside this function.
  model, x, c = _budgeted_msets_lp_sub(i, solver)
  set_normalized_rhs(c, β)
  optimize!(model)

  V = Array{Float64, 3}(undef, m(i), dimension(i) + 1, budget(i) + 1)
  S = Dict{Tuple{Int, Int, Int}, Vector{Int}}()

  if termination_status(model) == MOI.OPTIMAL
    sol = findall(JuMP.value.(x) .>= 0.5)
    V[m(i), 0 + 1, β + 1] = objective_value(model)
    S[m(i), 0, β] = sol
  else
    V[m(i), 0 + 1, β + 1] = -Inf
    S[m(i), 0, β] = Int[-1]
  end

  return BudgetedMSetSolution(i, S[m(i), 0, β], V, S)
end

function budgeted_msets_lp_select(i::BudgetedMSetInstance, budgets; solver=nothing)
  model, x, c = _budgeted_msets_lp_sub(i, solver)

  V = Array{Float64, 3}(undef, m(i), dimension(i) + 1, budget(i) + 1)
  S = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
  for budget in budgets
    set_normalized_rhs(c, budget)
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
      sol = findall(JuMP.value.(x) .>= 0.5)
      V[m(i), 0 + 1, budget + 1] = objective_value(model)
      S[m(i), 0, budget] = sol
    else
      V[m(i), 0 + 1, budget + 1] = -Inf
      S[m(i), 0, budget] = Int[-1]
    end
  end

  return BudgetedMSetSolution(i, S[m(i), 0, maximum(budgets)], V, S)
end

function budgeted_msets_lp_all(i::BudgetedMSetInstance; solver=nothing, max_budget::Int=budget(i))
  # Solve for all budgets at once, even though it is not really more efficient than looping outside this function.
  return budgeted_msets_lp_select(i, 0:max_budget, solver=solver)
end
