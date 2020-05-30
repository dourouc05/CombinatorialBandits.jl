solve(i::BipartiteMatchingInstance{T}, ::HungarianAlgorithm; kwargs...) where T = matching_hungarian(i; kwargs...)

function matching_hungarian(i::BipartiteMatchingInstance{T}) where T
  # No implementation of the Hungarian algorithm for matchings: reuse Munkres.jl.

  # First, build a reward matrix, as it's the basic level of knowledge of this package.
  # Use a roughly -\infty reward for edges that do not exist.
  reward_matrix = ones(i.n_left, i.n_right) * nextfloat(-Inf)
  for (k, v) in i.reward
    # idx1: always left. idx2: always right.
    if src(k) in i.vertex_left
      idx1 = findfirst(i.vertex_left .== src(k))
      idx2 = findfirst(i.vertex_right .== dst(k))
    else
      idx1 = findfirst(i.vertex_left .== dst(k))
      idx2 = findfirst(i.vertex_right .== src(k))
    end

    # Simple check for errors. This should only happen when the rewards dictionary has entries that do not match
    # edges in the (bipartite) graph.
    if isnothing(idx1) || isnothing(idx2)
      msg = "Indices not found for edge $k. Both ends of the edge belong to the "
      if ! isnothing(findfirst(i.vertex_left .== src(k))) && ! isnothing(findfirst(i.vertex_left .== dst(k)))
        msg *= "left"
      elseif ! isnothing(findfirst(i.vertex_right .== src(k))) && ! isnothing(findfirst(i.vertex_right .== dst(k)))
        msg *= "right"
      end
      msg *= " part of the bipartite graph!"

      # Ensure that execution is stopped now...
      error(msg)
    end

    reward_matrix[idx1, idx2] = v
  end

  # Make a cost matrix, to match the objective function of the underlying solver (Hungarian.jl).
  cost_matrix = maximum(reward_matrix) .- reward_matrix

  # Solve the corresponding matching problem. Discard the returned cost, as it makes no sense with the previous solution.
  sol, _ = Hungarian.hungarian(cost_matrix)

  # Return a list of edges in the graph, restoring the numbers of each vertex.
  solution = Edge{T}[]
  value = 0.0
  for (idx1, idx2) in enumerate(sol)
    # A 0 indicates that the vertex idx1 is not matched.
    if idx2 == 0
      continue
    end

    v1 = i.vertex_left[idx1]
    v2 = i.vertex_right[idx2]
    e = Edge(v1, v2)

    push!(solution, e)
    value += i.reward[e]
  end

  return BipartiteMatchingSolution(i, solution, value)
end