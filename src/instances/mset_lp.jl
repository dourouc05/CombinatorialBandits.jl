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

has_lp_formulation(::MSetLPSolver) = true
supports_solve_budgeted_linear(::MSetLPSolver) = true
supports_solve_all_budgeted_linear(::MSetLPSolver) = false
approximation_ratio(::MSetLPSolver) = 1.0
approximation_term(::MSetLPSolver) = 0.0
approximation_ratio_budgeted(::MSetLPSolver) = 1.0
approximation_term_budgeted(::MSetLPSolver) = 0.0

copy(solver::MSetLPSolver) = MSetLPSolver(solver.solver)

function build!(solver::MSetLPSolver, m::Int, n_arms::Int)
  solver.m = m
  solver.n_arms = n_arms
  solver.model = Model(solver.solver)
  solver.x = @variable(solver.model, [1:n_arms], binary=true, lower_bound=0.0)

  for i in 1:n_arms
    set_name(solver.x[i], "x_$i")
  end

  @constraint(solver.model, sum(solver.x) <= m)
end

function get_lp_formulation(solver::MSetLPSolver, rewards::Dict{T, Float64}) where T
  return solver.model,
    sum(rewards[i] * solver.x[i] for i in keys(rewards)),
    Dict{Int, JuMP.VariableRef}(i => solver.x[i] for i in keys(rewards))
end
