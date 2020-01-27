mutable struct MSetLPSolver <: MSetSolver
  solver
  m # Int
  n_arms # Int

  # Optimisation model (reused by solve_linear).
  model # ::Model
  x # ::Vector{Variable}

  function MSetLPSolver(solver)
    return new(solver, nothing, nothing, nothing, nothing)
  end
end

function build!(solver::MSetLPSolver, m::Int, n_arms::Int)
  solver.m = m
  solver.n_arms = n_arms
  solver.model = Model(solver.solver)
  solver.x = @variable(solver.model, [1:n_arms], binary=true)

  for i in 1:n_arms
    set_name(solver.x[i], "x_$i")
  end

  @constraint(solver.model, sum(solver.x) <= m)
end

has_lp_formulation(::MSetLPSolver) = true

function get_lp_formulation(solver::MSetLPSolver, rewards::Dict{T, Float64}) where T
  return solver.model,
    sum(rewards[i] * solver.x[i] for i in keys(rewards)),
    Dict{Int, JuMP.VariableRef}(i => solver.x[i] for i in keys(rewards))
end

function solve_linear(solver::MSetLPSolver, rewards::Dict{Int, Float64})
  m, obj, vars = get_lp_formulation(solver, rewards)
  @objective(m, Max, obj)

  # Maybe solve_budgeted_linear has already been called on this model.
  if :MSetLP in keys(m.ext)
    budget_constraint = m.ext[:MSetLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, 0)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Int[]
  end
  return Int[i for i in keys(rewards) if value(vars[i]) > 0.5]
end

function solve_budgeted_linear(solver::MSetLPSolver, rewards::Dict{Int, Float64}, weights::Dict{Int, Int}, budget::Int)
  m, obj, vars = get_lp_formulation(solver, rewards)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  if :MSetLP in keys(m.ext)
    budget_constraint = m.ext[:MSetLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weights[i] * solver.x[i] for i in keys(rewards)) >= budget)
    m.ext[:MSetLP] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Int[]
  end
  return Int[i for i in keys(rewards) if value(vars[i]) > 0.5]
end
