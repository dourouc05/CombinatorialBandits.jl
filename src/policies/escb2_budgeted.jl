struct ESCB2Budgeted <: ESCB2OptimisationAlgorithm
  ε::Float64 # Discretisation of the coefficients of the nonlinear part.
  solve_all_budgets_at_once::Bool # Some optimisation algorithms can solve the budgeted problems for all values of the budget at once.
  # TODO: compute a default value based on what the solver is able to do? Use Julia's applicable to determine it automatically?
end

function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, algo::ESCB2Budgeted,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64};
                                    with_trace::Bool=false) where T
  # Transform the linear term in a budget constraint, the nonlinear term
  # becoming the objective function (in which case the concave function
  # can be dropped).
  # For this to work, the linear part of the objective function must
  # take only integer values, i.e. the linear coefficients are actually
  # only integers.
  linear_discrete = Dict(k => round(Int, v / algo.ε, RoundUp) for (k, v) in linear)

  t0 = time_ns()

  # Maximum value of the linear term?
  max_budget = instance.n_arms * maximum(values(linear_discrete))

  # Solve the family of problems with an increasing budget.
  solutions = Dict{Int, Vector{T}}()
  if ! algo.solve_all_budgets_at_once # TODO: Replace this mandatory parameter by a function defined on the solvers, so that each of them can indicate what they support? Then, if this parameter is not set, the most efficient implementation is called if available; otherwise, follow this parameter (and still warn if the required implementation is not available).
    if ! applicable(solve_budgeted_linear, instance.solver, sqrtlinear, linear_discrete, max_budget)
      error("The function solve_budgeted_linear is not defined for the solver $(typeof(instance.solver)) for arguments of type ($(typeof(instance.solver)), $(typeof(sqrtlinear)), $(typeof(linear_discrete)), $(typeof(max_budget))).")
    end

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
  else
    if ! applicable(solve_all_budgeted_linear, instance.solver, sqrtlinear, linear_discrete, max_budget)
      error("The function solve_all_budgeted_linear is not defined for the solver $(typeof(instance.solver)) for arguments of type ($(typeof(instance.solver)), $(typeof(sqrtlinear)), $(typeof(linear_discrete)), $(typeof(max_budget))).")
    end

    solutions = solve_all_budgeted_linear(instance.solver, sqrtlinear, linear_discrete, max_budget)
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
