# Non-bipartite: https://theory.stanford.edu/~jvondrak/CS369P/lec5.pdf

mutable struct PerfectBipartiteMatchingLPSolver <: PerfectBipartiteMatchingSolver
  solver

  # Optimisation model (reused by solve_linear).
  model # ::Model
  x # ::Matrix{Variable}

  function PerfectBipartiteMatchingLPSolver(solver)
    return new(solver, nothing, nothing)
  end
end

has_lp_formulation(::PerfectBipartiteMatchingLPSolver) = true
supports_solve_budgeted_linear(::PerfectBipartiteMatchingLPSolver) = true
supports_solve_all_budgeted_linear(::PerfectBipartiteMatchingLPSolver) = false

function build!(solver::PerfectBipartiteMatchingLPSolver, n_arms::Int)
  # Build the optimisation model behind solve_linear.
  solver.model = Model(solver.solver)
  indices = collect((i, j) for i in 1:n_arms, j in 1:n_arms)
  solver.x = @variable(solver.model, [indices], binary=true)

  for i in 1:n_arms # Left nodes.
    @constraint(solver.model, sum(solver.x[(i, j)] for j in 1:n_arms) <= 1)
  end
  for j in 1:n_arms # Right nodes.
    @constraint(solver.model, sum(solver.x[(i, j)] for i in 1:n_arms) <= 1)
  end
end

function get_lp_formulation(solver::PerfectBipartiteMatchingLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  return solver.model,
    sum(reward[(i, j)] * solver.x[(i, j)] for (i, j) in keys(reward)),
    Dict{Tuple{Int, Int}, JuMP.VariableRef}((i, j) => solver.x[(i, j)] for (i, j) in keys(reward))
end

function solve_linear(solver::PerfectBipartiteMatchingLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Maybe solve_budgeted_linear has already been called on this model.
  if :PerfectBipartiteMatchingLP in keys(m.ext)
    budget_constraint = m.ext[:PerfectBipartiteMatchingLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, 0)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end
  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[(i, j)]) > 0.5]
end

function solve_budgeted_linear(solver::SpanningTreeLPSolver,
                               reward::Dict{Tuple{Int, Int}, Float64},
                               weight::Dict{Tuple{Int, Int}, T},
                               budget::Int) where {T<:Number} # Handle both Int and Float64
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  if :PerfectBipartiteMatchingLP in keys(m.ext)
    budget_constraint = m.ext[:PerfectBipartiteMatchingLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weight[i] * vars[i] for i in keys(reward)) >= budget)
    m.ext[:PerfectBipartiteMatchingLP] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end
  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5 || value(vars[j, i]) > 0.5]
end
