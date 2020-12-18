mutable struct PerfectBipartiteMatchingMunkresSolver <: PerfectBipartiteMatchingSolver
  weights_matrix::Matrix{Float64}

  function PerfectBipartiteMatchingMunkresSolver()
    return new(zeros(0, 0))
  end
end

supports_solve_budgeted_linear(::PerfectBipartiteMatchingMunkresSolver) = false
supports_solve_all_budgeted_linear(::PerfectBipartiteMatchingMunkresSolver) = false
has_lp_formulation(::PerfectBipartiteMatchingMunkresSolver) = false
approximation_ratio(::PerfectBipartiteMatchingMunkresSolver) = 1.0
approximation_term(::PerfectBipartiteMatchingMunkresSolver) = 0.0

function build!(solver::PerfectBipartiteMatchingMunkresSolver, n_arms::Int)
  solver.weights_matrix = zeros(n_arms, n_arms)
  nothing
end

function solve_linear(solver::PerfectBipartiteMatchingMunkresSolver, rewards::Dict{Tuple{Int, Int}, Float64})
  fill!(solver.weights_matrix, 0.0)
  for (k, v) in rewards
    solver.weights_matrix[k[1], k[2]] = v
  end
  solver.weights_matrix = maximum(solver.weights_matrix) .- solver.weights_matrix

  assignment = Munkres.munkres(solver.weights_matrix)
  return collect(enumerate(assignment))
end
