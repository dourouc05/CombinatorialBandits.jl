struct DynamicBudgetedBipartiteMatchingSolution{T, U} <: BudgetedBipartiteMatchingSolution{T, U}
  instance::BudgetedBipartiteMatchingInstance{T, U}
  solution::Vector{Edge{T}}
  value::Float64
  state::Dict{Tuple{Int, Int, Int, Int}, Float64} # Data structure built by the dynamic-programming recursion.
  solutions::Dict{Tuple{Int, Int, Int, Int}, Vector{Edge{T}}} # From the indices of state to the corresponding solution.
end

function matching_dp_budgeted(instance::BudgetedBipartiteMatchingInstance{T, U}) where {T, U}
  # First, determine the size of a perfect matching.
  nr = Dict(k => 1.0 for (k, v) in instance.matching.reward)
  ni = BipartiteMatchingInstance(instance.matching, nr)
  ns = matching_dp_imperfect(ni)
  m = length(ns.solution)

  # Solve this budgeted matching to ensure the constraints are met.

  # Four indices:
  # - two vertices, like matching_dp_imperfect;
  # - the number of edges still to add to have a perfect matching, like matching_dp;
  # - the minimum budget to respect.
  state = Dict{Tuple{Int, Int, Int, Int}, Float64}((i, j, k, l) => -Inf for i in instance.matching.vertex_left, j in instance.matching.vertex_right, k in 0:m, l in 0:instance.budget)
  solutions = Dict{Tuple{Int, Int, Int, Int}, Vector{Edge{T}}}((i, j, m, zero(U)) => Edge{T}[] for i in instance.matching.vertex_left, j in instance.matching.vertex_right)

  # Initialise: k=m-1, l=0, i.e. take at most one edge, don't consider the budget yet.
  v1 = first(instance.matching.vertex_right)
  for i in instance.matching.vertex_left
    if has_edge(instance.matching.graph, i, v1)
      state[i, v1, m - 1, 0] = instance.matching.reward[Edge(i, v1)]
      solutions[i, v1, m - 1, 0] = [Edge(i, v1)]
    end
  end

  v1 = first(instance.matching.vertex_left)
  for j in instance.matching.vertex_right
    if has_edge(instance.matching.graph, v1, j)
      state[v1, j, m - 1, 0] = instance.matching.reward[Edge(v1, j)]
      solutions[v1, j, m - 1, 0] = [Edge(v1, j)]
    end
  end

  for (i, prev_i) in zip(instance.matching.vertex_left[2:end], instance.matching.vertex_left[1:end-1])
    for (j, prev_j) in zip(instance.matching.vertex_right[2:end], instance.matching.vertex_right[1:end-1])
      # Take (i, j).
      if has_edge(instance.matching.graph, i, j)
        state[i, j, m - 1, 0] = instance.matching.reward[Edge(i, j)]
        solutions[i, j, m - 1, 0] = [Edge(i, j)]
      end

      # Don't take (i, j): copy a previous solution. No mixing possible, as only one edge is considered.
      if state[prev_i, j, m - 1, 0] > state[i, j, m - 1, 0]
        state[i, j, m - 1, 0] = state[prev_i, j, m - 1, 0]
        solutions[i, j, m - 1, 0] = solutions[prev_i, j, m - 1, 0]
      end
      if state[i, prev_j, m - 1, 0] > state[i, j, m - 1, 0]
        state[i, j, m - 1, 0] = state[i, prev_j, m - 1, 0]
        solutions[i, j, m - 1, 0] = solutions[i, prev_j, m - 1, 0]
      end
    end
  end

  # Initialisation for l=0, k < m-1: create perfect matchings, regardless of budget.
  for k in (m - 2):-1:0
    # Don't handle the cases with just one vertex in one part of the graph:
    # this loop requires at least two edges to be taken, but the considered
    # subset of edges can lead to at most only one being taken.

    # Consider the other vertices.
    for (i, prev_i) in zip(instance.matching.vertex_left[2:end], instance.matching.vertex_left[1:end-1])
      for (j, prev_j) in zip(instance.matching.vertex_right[2:end], instance.matching.vertex_right[1:end-1])
        # Take (i, j).
        if has_edge(instance.matching.graph, i, j) && state[prev_i, prev_j, k + 1, 0] > -Inf
          state[i, j, k, 0] = instance.matching.reward[Edge(i, j)] + state[prev_i, prev_j, k + 1, 0]
          solutions[i, j, k, 0] = solutions[prev_i, prev_j, k + 1, 0]
          push!(solutions[i, j, k, 0], Edge(i, j))
        end

        # Don't take (i, j): mix previous solutions (as a special case, maybe take a previous solution completely).
        for k2 in 0:m
          for k3 in 0:m
            if state[prev_i, j, k2, 0] + state[i, prev_j, k3, 0] > state[i, j, k, 0]
              potential_sol = union(Set{Edge{T}}(solutions[prev_i, j, k2, 0]), Set{Edge{T}}(solutions[i, prev_j, k3, 0]))
              val = sum(instance.matching.reward[Edge(src(e), dst(e))] for e in potential_sol)

              if length(potential_sol) == m - k && val > state[i, j, k, 0]
                state[i, j, k, 0] = val
                solutions[i, j, k, 0] = collect(potential_sol)
              end
            end
          end
        end
      end
    end
  end

  # Main iteration, considering the first vertex of each side is already done.
  for l in 1:instance.budget
    # Repeat the loop as often as necessary to handle the corner case with zero-weight edges.
    while true
      changes = false

      # Deal with the first two vertices (one on each side). All constraints are imposed.
      # Just one edge to be taken, as one vertex is always fixed on one side:
      # no feasible solution for k < m-1.
      vl = first(instance.matching.vertex_left)
      vr = first(instance.matching.vertex_right)
      if has_edge(instance.matching.graph, vl, vr) && instance.weight[Edge(vl, vr)] >= l
        state[vl, vr, m - 1, l] = instance.matching.reward[Edge(vl, vr)]
        solutions[vl, vr, m - 1, l] = [Edge(vl, vr)]
      end

      v1 = first(instance.matching.vertex_right)
      for i in instance.matching.vertex_left
        if has_edge(instance.matching.graph, i, v1) && instance.weight[Edge(i, v1)] >= l
          state[i, v1, m - 1, l] = instance.matching.reward[Edge(i, v1)]
          solutions[i, v1, m - 1, l] = [Edge(i, v1)]
        end
      end

      v1 = first(instance.matching.vertex_left)
      for j in instance.matching.vertex_right
        if has_edge(instance.matching.graph, v1, j) && instance.weight[Edge(v1, j)] >= l
          state[v1, j, m - 1, l] = instance.matching.reward[Edge(v1, j)]
          solutions[v1, j, m - 1, l] = [Edge(v1, j)]
        end
      end

      # Do the rest.
      for k in (m - 1):-1:0
        for (i, prev_i) in zip(instance.matching.vertex_left[2:end], instance.matching.vertex_left[1:end-1])
          for (j, prev_j) in zip(instance.matching.vertex_right[2:end], instance.matching.vertex_right[1:end-1])

            # Take (i, j) if possible.
            if has_edge(instance.matching.graph, i, j)
              remaining_budget = max(0, l - instance.weight[Edge(i, j)])

              if state[prev_i, prev_j, k + 1, remaining_budget] > -Inf
                state[i, j, k, l] = instance.matching.reward[Edge(i, j)] + state[prev_i, prev_j, k + 1, remaining_budget]
                solutions[i, j, k, l] = solutions[prev_i, prev_j, k + 1, remaining_budget]
                push!(solutions[i, j, k, l], Edge(i, j))

                changes = true
              end
            end

            # Don't take (i, j): mix previous solutions (as a special case, maybe take a previous solution completely).
            for k2 in 0:m
              for k3 in 0:m
                if state[prev_i, j, k2, l] + state[i, prev_j, k3, l] > state[i, j, k, l]
                  potential_sol = union(Set{Edge{T}}(solutions[prev_i, j, k2, l]), Set{Edge{T}}(solutions[i, prev_j, k3, l]))
                  val = sum(instance.matching.reward[Edge(src(e), dst(e))] for e in potential_sol)
                  wei = sum(instance.weight[Edge(src(e), dst(e))] for e in potential_sol)

                  if length(potential_sol) == m - k && val > state[i, j, k, l] && wei >= l
                    state[i, j, k, l] = val
                    solutions[i, j, k, l] = collect(potential_sol)

                    changes = true
                  end
                end
              end
            end
          end
        end
      end

      # If this iteration made no change, stop and go on to the next budget.
      if ! changes
        break
      end
    end
  end

  val = state[last(instance.matching.vertex_left), last(instance.matching.vertex_right), 0, instance.budget]
  sol = solutions[last(instance.matching.vertex_left), last(instance.matching.vertex_right), 0, instance.budget]

  return DynamicBudgetedBipartiteMatchingSolution(instance, sol, val, state, solutions)
  # return SimpleBudgetedBipartiteMatchingSolution(instance, sol)
end