function _maximise_nonlinear_through_budgeted(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, 
                                              weights::Dict{T, Int}, max_weight::Int, nl_func::Function, 
                                              solve_all_budgets_at_once::Bool) where T
  solutions = _maximise_nonlinear_through_budgeted_sub(instance, rewards, weights, max_weight, Val(solve_all_budgets_at_once))
  # TODO: Replace the mandatory parameter solve_all_budgets_at_once by a function defined on the solvers, so that each of them can indicate what they support? Then, if this parameter is not set, the most efficient implementation is called if available; otherwise, follow this parameter (and still warn if the required implementation is not available).
  # TODO: make it a trait of the underlying algorithm to get a default value. Do the same for ESCB2.

  best_solution = Int[]
  best_objective = -Inf
  for (budget, sol) in solutions
    # Ignore infeasible cases.
    if length(sol) == 0 || sol == [-1]
      continue
    end

    # Compute the maximum.
    f_x = nl_func(sol, budget)
    if f_x > best_objective
      best_solution = sol
      best_objective = f_x
    end
  end

  return best_solution, best_objective
end

function _maximise_nonlinear_through_budgeted_sub(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, 
                                                  weights::Dict{T, Int}, max_weight::Int, 
                                                  ::Val{true})::Dict{Int, Vector{T}} where T
  if ! applicable(solve_all_budgeted_linear, instance.solver, rewards, weights, max_weight)
    if applicable(solve_budgeted_linear, instance.solver, rewards, weights, max_weight)
      # @warn("The function solve_all_budgeted_linear is not defined for the solver $(typeof(instance.solver)).")
      return _maximise_nonlinear_through_budgeted_sub(instance, rewards, weights, max_weight, Val(false))
    else
      error("Neither solve_budgeted_linear nor solve_all_budgeted_linear are not defined for the solver $(typeof(instance.solver)).")
    end
  end

  return solve_all_budgeted_linear(instance.solver, rewards, weights, max_weight)
end

function _maximise_nonlinear_through_budgeted_sub(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, 
                                                  weights::Dict{T, Int}, max_weight::Int, 
                                                  ::Val{false})::Dict{Int, Vector{T}} where T
  if ! applicable(solve_budgeted_linear, instance.solver, rewards, weights, max_weight)
    if applicable(solve_all_budgeted_linear, instance.solver, rewards, weights, max_weight)
      # @warn("The function solve_budgeted_linear is not defined for the solver $(typeof(instance.solver)).")
      return _maximise_nonlinear_through_budgeted_sub(instance, rewards, weights, max_weight, Val(true))
    else
      error("Neither solve_budgeted_linear nor solve_all_budgeted_linear are not defined for the solver $(typeof(instance.solver)).")
    end
  end

  # Assumption: rewards is more complete than weights.
  @assert length(keys(rewards)) >= length(keys(weights))

  # Start the computation, memorising all solutions (one per value of the budget) along the way.
  solutions = Dict{Int, Vector{T}}()
  budget = 0
  while budget <= max_weight
    sol = solve_budgeted_linear(instance.solver, rewards, weights, budget)

    if length(sol) == 0 || sol == [-1]
      # Infeasible!
      for b in budget:max_weight
        solutions[b] = sol
      end
      break
    end

    # Feasible. Fill the dictionary as much as possible for this budget: if the constraint is not tight, the solution
    # is constant for all budgets between the value prescribed in the budget constraint until the obtained budget.
    sol_budget = sum(arm in keys(weights) ? weights[arm] : 0 for arm in keys(rewards) if arm in sol)
    for b in budget:sol_budget
      solutions[b] = sol
    end
    budget = sol_budget + 1
  end

  return solutions
end
