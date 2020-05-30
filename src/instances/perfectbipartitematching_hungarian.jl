mutable struct PerfectBipartiteMatchingHungarianSolver <: PerfectBipartiteMatchingSolver
  weights_matrix::Matrix{Float64}

  function PerfectBipartiteMatchingHungarianSolver()
    return new(zeros(0, 0))
  end
end

supports_solve_budgeted_linear(::PerfectBipartiteMatchingHungarianSolver) = false
supports_solve_all_budgeted_linear(::PerfectBipartiteMatchingHungarianSolver) = false

function build!(solver::PerfectBipartiteMatchingHungarianSolver, n_arms::Int)
  solver.weights_matrix = zeros(n_arms, n_arms)
  nothing
end

function solve_linear(solver::PerfectBipartiteMatchingHungarianSolver, rewards::Dict{Tuple{Int, Int}, Float64})
  fill!(solver.weights_matrix, 0.0)
  for (k, v) in rewards
    solver.weights_matrix[k[1], k[2]] = v
  end
  solver.weights_matrix = maximum(solver.weights_matrix) .- solver.weights_matrix

  matching = Hungarian.munkres(solver.weights_matrix)
  indices = CartesianIndices(matching)
  return [(indices[v][1], indices[v][2]) for v in findall(matching .== Hungarian.STAR)]
end

has_lp_formulation(::PerfectBipartiteMatchingHungarianSolver) = false
approximation_ratio(::PerfectBipartiteMatchingHungarianSolver) = 1.0
approximation_term(::PerfectBipartiteMatchingHungarianSolver) = 0.0
