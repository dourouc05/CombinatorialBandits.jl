solve(i::BipartiteMatchingInstance{T}, ::DynamicProgramming; kwargs...) where T = matching_dp(i; kwargs...)

function matching_dp_imperfect(instance::BipartiteMatchingInstance{T}) where T
  state = Dict{Tuple{Int, Int}, Float64}((i, j) => -Inf for i in instance.vertex_left, j in instance.vertex_right)
  solutions = Dict{Tuple{Int, Int}, Vector{Edge{T}}}()

  # Initialise: j=1 and i=1.
  v1 = first(instance.vertex_right)
  for i in instance.vertex_left
    if has_edge(instance.graph, i, v1)
      state[i, v1] = instance.reward[Edge(i, v1)]
      solutions[i, v1] = [Edge(i, v1)]
    end
  end

  v1 = first(instance.vertex_left)
  for j in instance.vertex_right
    if has_edge(instance.graph, v1, j)
      state[v1, j] = instance.reward[Edge(v1, j)]
      solutions[v1, j] = [Edge(v1, j)]
    end
  end

  # Main iteration, considering the first vertex of each side is already done.
  for (i, prev_i) in zip(instance.vertex_left[2:end], instance.vertex_left[1:end-1])
    for (j, prev_j) in zip(instance.vertex_right[2:end], instance.vertex_right[1:end-1])
      # Take (i, j).
      if has_edge(instance.graph, i, j) && state[prev_i, prev_j] > -Inf
        state[i, j] = instance.reward[Edge(i, j)] + state[prev_i, prev_j]
        solutions[i, j] = solutions[prev_i, prev_j]
        push!(solutions[i, j], Edge(i, j))
      end

      # Don't take (i, j): copy a previous solution.
      if state[prev_i, j] > state[i, j]
        state[i, j] = state[prev_i, j]
        solutions[i, j] = solutions[prev_i, j]
      end
      if state[i, prev_j] > state[i, j]
        state[i, j] = state[i, prev_j]
        solutions[i, j] = solutions[i, prev_j]
      end
    end
  end

  val = state[last(instance.vertex_left), last(instance.vertex_right)]
  sol = solutions[last(instance.vertex_left), last(instance.vertex_right)]

  return BipartiteMatchingSolution(instance, sol, val)
  # return state, solutions
end

function matching_dp(instance::BipartiteMatchingInstance{T}) where T
  # First, determine the size of a perfect matching.
  nr = Dict(k => 1.0 for (k, v) in instance.reward)
  ni = BipartiteMatchingInstance(instance, nr)
  ns = matching_dp_imperfect(ni)
  m = length(ns.solution)

  # If solving the matching as imperfect gives the right size, then done.
  naive_sol = matching_dp_imperfect(instance)
  if length(naive_sol.solution) == m
    return naive_sol
  end

  # Otherwise, solve this matching to ensure the perfect constraint is met.
  # TODO: reuse the state from the previous step?

  # Three indices: two vertices, like matching_dp_imperfect; the number of edges still to add to have a perfect matching.
  state = Dict{Tuple{Int, Int, Int}, Float64}((i, j, k) => -Inf for i in instance.vertex_left, j in instance.vertex_right, k in 0:m)
  solutions = Dict{Tuple{Int, Int, Int}, Vector{Edge{T}}}((i, j, m) => Edge{T}[] for i in instance.vertex_left, j in instance.vertex_right)

  # Initialise: j=1 and i=1, k=m-1.
  v1 = first(instance.vertex_right)
  for i in instance.vertex_left
    if has_edge(instance.graph, i, v1)
      state[i, v1, m - 1] = instance.reward[Edge(i, v1)]
      solutions[i, v1, m - 1] = [Edge(i, v1)]
    end
  end

  v1 = first(instance.vertex_left)
  for j in instance.vertex_right
    if has_edge(instance.graph, v1, j)
      state[v1, j, m - 1] = instance.reward[Edge(v1, j)]
      solutions[v1, j, m - 1] = [Edge(v1, j)]
    end
  end

  # Main iteration, considering the first vertex of each side is already done.
  for k in (m - 2):-1:0
    for (i, prev_i) in zip(instance.vertex_left[2:end], instance.vertex_left[1:end-1])
      for (j, prev_j) in zip(instance.vertex_right[2:end], instance.vertex_right[1:end-1])
        # Take (i, j).
        if has_edge(instance.graph, i, j) && state[prev_i, prev_j, k + 1] > -Inf
          state[i, j, k] = instance.reward[Edge(i, j)] + state[prev_i, prev_j, k + 1]
          solutions[i, j, k] = solutions[prev_i, prev_j, k + 1]
          push!(solutions[i, j, k], Edge(i, j))
        end

        # Don't take (i, j): mix previous solutions (as a special case, maybe take a previous solution completely).
        for k2 in 0:m
          for k3 in 0:m
            if state[prev_i, j, k2] + state[i, prev_j, k3] > state[i, j, k]
              potential_sol = union(Set{Edge{T}}(solutions[prev_i, j, k2]), Set{Edge{T}}(solutions[i, prev_j, k3]))
              val = sum(instance.reward[Edge(src(e), dst(e))] for e in potential_sol)

              if length(potential_sol) == m - k && val > state[i, j, k]
                state[i, j, k] = val
                solutions[i, j, k] = collect(potential_sol)
              end
            end
          end
        end
      end
    end
  end

  val = state[last(instance.vertex_left), last(instance.vertex_right), 0]
  sol = solutions[last(instance.vertex_left), last(instance.vertex_right), 0]

  return BipartiteMatchingSolution(instance, sol, val)
end
