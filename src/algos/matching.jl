struct BipartiteMatchingInstance{T}
  graph::AbstractGraph{T} # Ensured to be bipartite.
  reward::Dict{Edge{T}, Float64}

  partition::Vector{UInt8}
  n_left::Int
  n_right::Int
  vertex_left::Vector{T}
  vertex_right::Vector{T}

  function BipartiteMatchingInstance(graph::AbstractGraph{T}, reward::Dict{Edge{T}, Float64}) where T
    # Implicit assumption: all edges in the rewards exist in the graph; all edges in the graph have a reward.

    bmap = bipartite_map(graph)
    if length(bmap) != nv(graph) # Condition from is_bipartite
      error("The input graph is not bipartite!")
    end

    n_left = sum(bmap .== 1)
    n_right = sum(bmap .== 2)

    emap = collect(enumerate(bmap))
    emap1 = collect(filter(v -> v[2] == 1, emap))
    emap2 = collect(filter(v -> v[2] == 2, emap))

    vertex_left = map(first, emap1)
    vertex_right = map(first, emap2)

    @assert length(vertex_left) == n_left
    @assert length(vertex_right) == n_right
    @assert n_left + n_right == nv(graph)

    return new{T}(graph, reward, bmap, n_left, n_right, vertex_left, vertex_right)
  end

  function BipartiteMatchingInstance(i::BipartiteMatchingInstance{T}, reward::Dict{Edge{T}, Float64}) where T
    # Only for private use. Change the rewards of a given instance.
    return new{T}(i.graph, reward, i.partition, i.n_left, i.n_right, i.vertex_left, i.vertex_right)
  end
end

struct BipartiteMatchingSolution{T}
  instance::BipartiteMatchingInstance{T}
  solution::Vector{Edge{T}}
  value::Float64
end

function matching_hungarian(i::BipartiteMatchingInstance{T}) where T
  # No implementation of the Hungarian algorithm for matchings: reuse Munkres.jl.
  reward_matrix = zeros(i.n_left, i.n_right)
  for (k, v) in i.reward
    # idx1: always left. idx2: always right.
    if src(k) in i.vertex_left
      idx1 = findfirst(i.vertex_left .== src(k))
      idx2 = findfirst(i.vertex_right .== dst(k))
    else
      idx1 = findfirst(i.vertex_left .== dst(k))
      idx2 = findfirst(i.vertex_right .== src(k))
    end

    reward_matrix[idx1, idx2] = v
  end

  # Make a cost matrix, to match the objective function of the underlying solver.
  cost_matrix = maximum(reward_matrix) .- reward_matrix

  # Solve the corresponding matching problem. Discard the returned cost, as it makes no sense with the previous solution.
  sol, _ = Hungarian.hungarian(cost_matrix)

  # Return a list of edges in the graph, restoring the numbers of each vertex.
  solution = Edge{T}[]
  value = 0.0
  for (idx1, idx2) in enumerate(sol)
    v1 = i.vertex_left[idx1]
    v2 = i.vertex_right[idx2]
    e = Edge(v1, v2)

    push!(solution, e)
    value += i.reward[e]
  end

  return BipartiteMatchingSolution(i, solution, value)
end
