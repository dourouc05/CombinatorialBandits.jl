# Minimum spanning tree in complete graphs.
# Only the upper part of the reward matrix is considered!

abstract type SpanningTreeSolver end
function build!(solver::SpanningTreeSolver, graph::SimpleGraph)
  nothing
end

struct SpanningTree <: CombinatorialInstance{Tuple{Int, Int}}
  # Mandatory properties.
  n_arms::Int
  optimal_average_reward::Float64

  # Probability distributions for the arm rewards.
  graph::SimpleGraph
  reward::Dict{Tuple{Int, Int}, Distribution} # Roughly a matrix, except that not all entries should be filled.
  # Even for a complete graph, the matrix has too many elements: the diagonal is not used.
  # For a very large number of reads into the data structure, the dictionary seems a good fit.
  # TODO: uniformise reward vs. rewards.

  # Internal solver.
  solver::SpanningTreeSolver

  function SpanningTree(graph::SimpleGraph, reward::Dict{Tuple{Int, Int}, Distribution}, solver::SpanningTreeSolver)
    if length(reward) != ne(graph)
      error("The edges and rewards do not perfectly match: there are $(length(reward)) rewards and $(ne(graph)) edges.")
    end
    for v in keys(reward)
      if ! has_edge(graph, v[1], v[2])
        error("The edges and rewards do not perfectly match: the edge $v has a reward, but does not exist in the graph.")
      end
    end

    # Prepare the solver to be used (if required).
    build!(solver, graph)

    avg_reward = Dict{Tuple{Int, Int}, Float64}(k => mean(v) for (k, v) in reward)
    opt_sol = solve_linear(solver, avg_reward)
    opt = sum(avg_reward[i] for i in opt_sol)

    # Done!
    return new(ne(graph), opt, graph, reward, solver)
  end
end

function initial_state(instance::SpanningTree)
  zero_counts = Dict(k => 0 for (k, _) in instance.reward)
  zero_rewards = Dict(k => 0.0 for (k, _) in instance.reward)
  return State{Tuple{Int, Int}}(0, 0.0, 0.0, zero_counts, zero_rewards, copy(zero_rewards))
end

solve_linear(instance::SpanningTree, rewards::Dict{Tuple{Int, Int}, Float64}) = solve_linear(instance.solver, rewards)
has_lp_formulation(instance::SpanningTree) = has_lp_formulation(instance.solver)
get_lp_formulation(instance::SpanningTree, rewards::Dict{Tuple{Int, Int}, Float64}) = has_lp_formulation(instance) ?
  get_lp_formulation(instance.solver, rewards) :
  error("The complete graph minimum spanning tree solver uses no LP formulation.")

function _mst_solution_normalise(reward::Dict{Tuple{Int, Int}, Float64}, solution::Vector{Tuple{Int, Int}})
  # Not all solvers return the right kind of edges: they are sometimes reversed with respect
  # to the input graph. Hence return those so that all arms in the output solution
  # correspond exactly to those in the input graph. 
  normalised = Tuple{Int, Int}[]
  for e in solution
    if e in keys(reward)
      push!(normalised, e)
    else
      push!(normalised, (e[2], e[1]))
    end
  end
  return normalised
end

_mst_solution_normalise(instance::SpanningTree, solution::Vector{Tuple{Int, Int}}) =
  _mst_solution_normalise(instance.reward, solution)

function is_feasible(instance::SpanningTree, arms::Vector{Tuple{Int, Int}})
  n = nv(instance.graph)
  if length(arms) != n - 1
    return false
  end

  # An MST is feasible if all nodes are reachable from any given node (undirected graph).
  # However, not all edges are in order in the input vector.
  reachable_nodes = BitSet()
  sizehint!(reachable_nodes, n)
  push!(reachable_nodes, arms[1][1])

  arms_to_do = BitSet(1:(n - 1))

  while ! isempty(arms_to_do)
    has_changed = false

    for arm_id in arms_to_do
      arm = arms[arm_id]
      if arm[1] ∈ reachable_nodes
        push!(reachable_nodes, arm[2])
        delete!(arms_to_do, arm_id)
        has_changed = true
      end

      if arm[2] ∈ reachable_nodes
        push!(reachable_nodes, arm[1])
        delete!(arms_to_do, arm_id)
        has_changed = true
      end
    end

    if ! has_changed
      return false
    end
  end

  return true
end

function is_partially_acceptable(instance::SpanningTree, arms::Vector{Tuple{Int, Int}})
  if length(arms) == 0
    return true
  end

  if length(arms) > nv(instance.graph) - 1
    return false
  end

  # An MST is partially acceptable if it contains no cycle. However, not all edges are in order in the input vector.
  # Implement this with a graph search: if a previous node is found again (i.e. both ends of an edge have already been visited),
  # then there is a cycle.
  reached_nodes = BitSet()
  sizehint!(reached_nodes, length(arms) + 1)

  push!(reached_nodes, arms[1][1])
  push!(reached_nodes, arms[1][2])
  for i in 2:length(arms)
    arm1in = arms[i][1] ∈ reached_nodes
    arm2in = arms[i][2] ∈ reached_nodes

    if arm1in && arm2in
      return false
    end

    push!(reached_nodes, arms[i][1])
    push!(reached_nodes, arms[i][2])
  end

  return true
end
