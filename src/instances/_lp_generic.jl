_generic_lp_key(solver) = Symbol(string(typeof(solver)))

function solve_linear(solver, reward::Dict{T, Float64}) where T
  # Generic implementation that should work for all solvers following the same conventions:
  #  - `has_lp_formulation` returns `true` for the solver
  #  - `supports_solve_budgeted_linear` returns `true` for the solver
  #  - `get_lp_formulation` returns a LP-like formulation (may even be a MIP)
  # Lazy formulations are allowed.
  @assert has_lp_formulation(solver)
  @assert supports_solve_budgeted_linear(solver)

  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Maybe solve_budgeted_linear has already been called on this model.
  key = _generic_lp_key(solver)
  if key in keys(m.ext)
    budget_constraint = m.ext[key][:budget_constraint]
    set_normalized_rhs(budget_constraint, 0)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return T[]
  end
  return T[arm for arm in keys(reward) if value(vars[arm]) > 0.5]
end

function solve_budgeted_linear(solver,
                               reward::Dict{T, Float64},
                               weight::Dict{T, S},
                               budget::Int) where {T, S<:Number} # Handle both Int and Float64
  # Generic implementation that should work for all solvers following the same conventions:
  #  - `has_lp_formulation` returns `true` for the solver
  #  - `supports_solve_budgeted_linear` returns `true` for the solver
  #  - `get_lp_formulation` returns a LP-like formulation (may even be a MIP)
  #    whose variables are alse used for the budget constraint
  # Lazy formulations are allowed. 
  @assert has_lp_formulation(solver)
  @assert supports_solve_budgeted_linear(solver)

  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  key = _generic_lp_key(solver)
  if key in keys(m.ext)
    budget_constraint = m.ext[key][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weight[arm] * vars[arm] for arm in keys(reward)) >= budget)
    m.ext[key] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return T[]
  end
  return T[arm for arm in keys(reward) if value(vars[arm]) > 0.5]
end
