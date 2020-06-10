struct ESCB2Budgeted <: ESCB2OptimisationAlgorithm
  ε::Union{Float64, Nothing} # Discretisation of the coefficients of the nonlinear part, if forced by the user.
  solve_all_budgets_at_once::Union{Bool, Nothing} # Some optimisation algorithms can solve the budgeted problems for all values of the budget at once more efficiently.
end

ESCB2Budgeted() = ESCB2Budgeted(nothing, nothing)
ESCB2Budgeted(ε::Float64) = ESCB2Budgeted(ε, nothing)
ESCB2Budgeted(s::Bool) = ESCB2Budgeted(nothing, s)

function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, algo::ESCB2Budgeted,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64}, bandit_round::Int;
                                    with_trace::Bool=false) where T
  # Transform the linear term in a budget constraint, the nonlinear term
  # becoming the objective function (in which case the concave function
  # can be dropped).
  # For this to work, the linear part of the objective function must
  # take only integer values, i.e. the linear coefficients are actually
  # only integers.
  ξ = if algo.ε !== nothing
    algo.ε
  else
    logm = max(1.0, log(instance.m))
    δ = length(linear) * (logm ^ 2) / bandit_round
    δ / instance.m
  end
  linear_discrete = Dict(k => round(Int, v / ξ, RoundUp) for (k, v) in linear)

  # Fill automatic values.
  if algo.solve_all_budgets_at_once === nothing
    algo = copy(algo) # Don't modify the user's object.
    algo.solve_all_budgets_at_once = supports_solve_all_budgeted_linear(instance)
  end

  # Start timing.
  t0 = time_ns()

  # Maximum value of the linear term?
  max_budget = instance.n_arms * maximum(values(linear_discrete))

  # Solve the family of problems with an increasing budget.
  solutions = Dict{Int, Vector{T}}()
  if supports_solve_budgeted_linear(instance) && ! algo.solve_all_budgets_at_once
    budget = 0
    while budget <= max_budget
      sol = solve_budgeted_linear(instance.solver, sqrtlinear, linear_discrete, budget)

      if length(sol) == 0 || sol == [-1]
        # Infeasible!
        for b in budget:max_budget
          solutions[b] = sol
        end
        break
      end

      # Feasible.
      sol_budget = sum(linear_discrete[arm] for arm in keys(linear_discrete) if arm in sol)
      for b in budget:sol_budget
        solutions[b] = sol
      end
      budget = sol_budget + 1
    end
  elseif supports_solve_all_budgeted_linear(instance)
    solutions = solve_all_budgeted_linear(instance.solver, sqrtlinear, linear_discrete, max_budget)
  else
    error("At least one of the functions solve_all_budgeted_linear and solve_budgeted_linear " *
          "is not defined for the solver " *
          "$(typeof(instance.solver)) for arguments of type ($(typeof(instance.solver)), $(typeof(sqrtlinear)), " *
          "$(typeof(linear_discrete)), $(typeof(max_budget))). This function is required for this implementation of ESCB2.")
  end

  # Take the best solution.
  best_solution = Int[]
  best_objective = -Inf
  for (budget, sol) in solutions
    # Ignore infeasible cases.
    if length(sol) == 0 || sol == [-1]
      continue
    end

    if sum(sqrtlinear[arm] for arm in keys(sqrtlinear) if arm in sol) < 0.0
      continue
    end

    # Compute the maximum.
    reward = escb2_index(linear, sqrtlinear, sol)
    if reward > best_objective
      best_solution = sol
      best_objective = reward
    end
  end
  t1 = time_ns()

  if with_trace
    run_details = ESCB2Details()
    run_details.n_iterations = 1
    run_details.best_objective = best_objective
    run_details.solver_time = (t1 - t0) / 1_000_000_000
    return best_solution, run_details
  else
    return best_solution
  end
end
