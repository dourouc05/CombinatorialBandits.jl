# Single-source shortest path in complete graphs.
# Only the upper part of the reward matrix is considered!

abstract type ElementaryPathSolver end
function build!(solver::ElementaryPathSolver, graph::SimpleDiGraph, source::Int, destination::Int)
  nothing
end

struct ElementaryPath <: CombinatorialInstance{Tuple{Int, Int}}
  # Mandatory properties.
  n_arms::Int
  optimal_average_reward::Float64

  # Probability distributions for the arm rewards.
  graph::SimpleDiGraph
  reward::Dict{Tuple{Int, Int}, Distribution} # Roughly a matrix, except that not all entries should be filled.
  # Even for a complete graph, the matrix has too many elements: the diagonal is not used.
  # For a very large number of reads into the data structure, the dictionary seems a good fit.

  # Characteristics of the path.
  source::Int
  destination::Int

  # Internal solver.
  solver::ElementaryPathSolver

  function ElementaryPath(graph::SimpleDiGraph, reward::Dict{Tuple{Int, Int}, <:Distribution}, source::Int, destination::Int, solver::ElementaryPathSolver)
    n = nv(graph)

    if source <= 0
      error("Source node is not an acceptable node index: $(source) is negative or zero. The first node is numbered 1.")
    end
    if source > n
      error("Source node is not an acceptable node index: $(source) is too large. The last node is numbered $n.")
    end
    if destination <= 0
      error("Destination node is not an acceptable node index: $(destination) negative or zero. The first node is numbered 1.")
    end
    if destination > n
      error("Destination node is not an acceptable node index: $(destination) is too large. The last node is numbered $n.")
    end

    if length(reward) != ne(graph)
      error("The edges and rewards do not perfectly match: there are $(length(reward)) rewards and $(ne(graph)) edges.")
    end
    for v in keys(reward)
      if ! has_edge(graph, v[1], v[2])
        error("The edges and rewards do not perfectly match: the edge $v has a reward, but does not exist in the graph.")
      end
    end

    if is_cyclic(graph)
      @warn "The input graph is cyclic, bandit algorithms are usually not useful in this case"
      # Just play a positive-reward cycle an infinite number of times at each round...
    end

    # Prepare the solver to be used (if required).
    build!(solver, graph, source, destination)

    avg_reward = Dict{Tuple{Int, Int}, Float64}(k => mean(v) for (k, v) in reward)
    opt_sol = solve_linear(solver, avg_reward)
    opt = sum(avg_reward[i] for i in opt_sol)

    # Done!
    return new(ne(graph), opt, graph, reward, source, destination, solver)
  end
end

Base.copy(instance::ElementaryPath) = ElementaryPath(instance.graph, instance.reward, instance.source, instance.destination, copy(instance.solver))

function is_feasible(instance::ElementaryPath, arms::Vector{Tuple{Int, Int}})
  # Many checks are similar for just a partially acceptable path.
  if ! is_partially_acceptable(instance, arms)
    return false
  end

  if length(arms) == 0
    return false
  end

  # Check whether the path goes to the destination. Think about the case where
  # the path is just one edge. The check about the source is already performed
  # by is_partially_acceptable.
  last_edge = arms[end]
  if last_edge[2] != instance.destination
    return false
  end

  return true
end

function is_partially_acceptable(instance::ElementaryPath, arms::Vector{Tuple{Int, Int}})
  path_length = length(arms)

  if path_length > instance.n_arms
    return false
  end

  if path_length < 1 # Major difference with is_feasible: an empty path is partially acceptable!
    return true
  end

  n = nv(instance.graph)

  # Check whether the path starts from the source and possibly ends at the destination.
  # Think about the case where the path is just one edge.
  for (i, arm) in enumerate(arms)
    # The first edge must start at the source.
    if i == 1 && arm[1] != instance.source
      return false
    end

    # No edge can end at the destination if it is not the last one in the path (otherwise, cycle).
    if i < length(arms) && arm[2] == instance.destination
      return false
    end

    # The last edge must end at the destination, but only if the path is as long as possible.
    if i == n && path_length == instance.n_arms && arm[2] != instance.destination
      return false
    end

    # If after the first edge, this edge must be adjacent to the same node as the previous one.
    if i > 1 && arms[i - 1][2] != arm[1]
      return false
    end
  end

  return true
end
