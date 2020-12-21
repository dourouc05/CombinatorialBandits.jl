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
approximation_ratio(::PerfectBipartiteMatchingLPSolver) = 1.0
approximation_term(::PerfectBipartiteMatchingLPSolver) = 0.0
approximation_ratio_budgeted(::PerfectBipartiteMatchingLPSolver) = 1.0
approximation_term_budgeted(::PerfectBipartiteMatchingLPSolver) = 0.0
supports_solve_budgeted_linear(::PerfectBipartiteMatchingLPSolver) = true
supports_solve_all_budgeted_linear(::PerfectBipartiteMatchingLPSolver) = false

Base.copy(solver::PerfectBipartiteMatchingLPSolver) = PerfectBipartiteMatchingLPSolver(solver.solver)

function build!(solver::PerfectBipartiteMatchingLPSolver, n_arms::Int)
  # Build the optimisation model behind solve_linear.
  solver.model = Model(solver.solver)
  indices = collect((i, j) for i in 1:n_arms, j in 1:n_arms)
  solver.x = @variable(solver.model, [indices], binary=true)

  for i in 1:n_arms
    for j in 1:n_arms
      set_name(solver.x[(i, j)], "x[$i, $j]")
    end
  end

  for i in 1:n_arms # Left nodes.
    @constraint(solver.model, sum(solver.x[(i, j)] for j in 1:n_arms) == 1)
  end
  for j in 1:n_arms # Right nodes.
    @constraint(solver.model, sum(solver.x[(i, j)] for i in 1:n_arms) == 1)
  end
end

function get_lp_formulation(solver::PerfectBipartiteMatchingLPSolver, rewards::Dict{Tuple{Int, Int}, Float64})
  return solver.model,
    sum(rewards[(i, j)] * solver.x[(i, j)] for (i, j) in keys(rewards)),
    Dict{Tuple{Int, Int}, JuMP.VariableRef}((i, j) => solver.x[(i, j)] for (i, j) in keys(rewards))
end
