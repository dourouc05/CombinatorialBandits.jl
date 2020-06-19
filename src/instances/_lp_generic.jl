function solve_linear(solver, reward::Dict{Tuple{Int, Int}, Float64})
  # Generic implementation that should work for all solvers following the same conventions:
  #  - `has_lp_formulation` returns `true` for the solver
  #  - `supports_solve_budgeted_linear` returns `true` for the solver
  #  - `get_lp_formulation` returns a LP-like formulation (may even be a MIP)
  # Lazy formulations are allowed.
  @assert has_lp_formulation(solver)
  @assert supports_solve_budgeted_linear(solver)

  key = Symbol(string(typeof(solver)))

  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Maybe solve_budgeted_linear has already been called on this model.
  if key in keys(m.ext)
    budget_constraint = m.ext[key][:budget_constraint]
    set_normalized_rhs(budget_constraint, 0)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end

  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5 || value(vars[j, i]) > 0.5]
end

function solve_budgeted_linear(solver,
                               reward::Dict{Tuple{Int, Int}, Float64},
                               weight::Dict{Tuple{Int, Int}, T},
                               budget::Int) where {T<:Number} # Handle both Int and Float64
  # Generic implementation that should work for all solvers following the same conventions:
  #  - `has_lp_formulation` returns `true` for the solver
  #  - `supports_solve_budgeted_linear` returns `true` for the solver
  #  - `get_lp_formulation` returns a LP-like formulation (may even be a MIP)
  #    whose variables are alse used for the budget constraint
  # Lazy formulations are allowed. 
  @assert has_lp_formulation(solver)
  @assert supports_solve_budgeted_linear(solver)

  key = Symbol(string(typeof(solver)))

  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  if key in keys(m.ext)
    budget_constraint = m.ext[key][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weight[i] * vars[i] for i in keys(reward)) >= budget)
    m.ext[key] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end
  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5 || value(vars[j, i]) > 0.5]
end
