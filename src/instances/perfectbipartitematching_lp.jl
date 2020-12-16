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
