mutable struct SpanningTreeLightGraphsPrimSolver <: SpanningTreeSolver
  graph::SimpleGraph
  weights_matrix::Matrix{Float64}

  function SpanningTreeLightGraphsPrimSolver()
    return new(Graph(0), zeros(0, 0))
  end
end

function build!(solver::SpanningTreeLightGraphsPrimSolver, graph::SimpleGraph)
  n = nv(graph)
  solver.graph = graph
  solver.weights_matrix = zeros(n, n)
end

function solve_linear(solver::SpanningTreeLightGraphsPrimSolver, reward::Dict{Tuple{Int, Int}, Float64})
  # Make up a rewards matrix by copying the input dictionary into the right data structure for the shortest path computations.
  fill!(solver.weights_matrix, 0.0)
  for (k, v) in reward
    solver.weights_matrix[k[1], k[2]] = v
  end
  solver.weights_matrix = maximum(solver.weights_matrix) .- solver.weights_matrix

  # Compute the maximum spanning tree.
  mst = prim_mst(solver.graph, solver.weights_matrix)

  # Transform edges into tuples.
  return _mst_solution_normalise(reward, [(src(e), dst(e)) for e in mst])
end

has_lp_formulation(::SpanningTreeLightGraphsPrimSolver) = false
