mutable struct ElementaryPathLightGraphsDijkstraSolver <: ElementaryPathSolver
  graph::SimpleDiGraph
  weights_matrix::Matrix{Float64}
  source::Int
  destination::Int

  function ElementaryPathLightGraphsDijkstraSolver()
    return new(DiGraph(0), zeros(0, 0), -1, -1)
  end
end

function build!(solver::ElementaryPathLightGraphsDijkstraSolver, graph::SimpleDiGraph, source::Int, destination::Int)
  n = nv(graph)
  solver.graph = graph
  solver.weights_matrix = zeros(n, n)
  solver.source = source
  solver.destination = destination
end

has_lp_formulation(::ElementaryPathLightGraphsDijkstraSolver) = false

function solve_linear(solver::ElementaryPathLightGraphsDijkstraSolver, rewards::Dict{Tuple{Int, Int}, Float64})
  # Make up a rewards matrix by copying the input dictionary into the right data structure for the shortest path computations.
  fill!(solver.weights_matrix, 0.0)
  for (k, v) in rewards
    solver.weights_matrix[k[1], k[2]] = v
  end
  solver.weights_matrix = maximum(solver.weights_matrix) .- solver.weights_matrix

  # Compute the shortest path.
  state = dijkstra_shortest_paths(solver.graph, solver.source, solver.weights_matrix)

  # Retrieve the shortest path based on the returned data structure (using the predecessor node).
  # Everything must be done in reverse, due to the way the path is stored for Dijkstra.
  path = Vector{Tuple{Int, Int}}()
  current_node = solver.destination
  while current_node != solver.source
    push!(path, (state.parents[current_node][1], current_node))
    current_node = state.parents[current_node][1]
  end
  return path
end
