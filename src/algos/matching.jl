struct BipartiteMatchingInstance{T}
  graph::AbstractGraph{T} # Ensured to be bipartite.
  rewards::Dict{Edge{T}, Float64}

  partition::Vector{UInt8}
  n_left::Int
  n_right::Int
  id_to_vertex_left::Dict{Int, Edge{T}}
  id_to_vertex_right::Dict{Int, Edge{T}}
  vertex_to_id_left::Vector{Edge{T}, Int}
  vertex_to_id_right::Vector{Edge{T}, Int}

  function BipartiteMatchingInstance(graph::AbstractGraph{T}, rewards::Dict{Edge{T}, Float64}) where T
    map = bipartite_map(graph)
    if length(map) != nv(g) # Condition from is_bipartite
      error("The input graph is not bipartite!")
    end

    n_left = sum(map .== 1)
    n_right = sum(map .== 2)
    id_to_vertex_left = collect(filter(v -> v[2] == 1, collect(enumerate(map))))
    id_to_vertex_right = collect(filter(v -> v[2] == 2, collect(enumerate(map))))
    vertex_to_id_left = Dict(v => k for (k, v) in id_to_vertex_left)
    vertex_to_id_right = Dict(v => k for (k, v) in id_to_vertex_right)

    @assert length(vertices_left) == n_left
    @assert length(vertices_right) == n_right
    @assert n_left + n_right == nv(graph)

    return new(graph, map, rewards, n_left, n_right, id_to_vertex_left, id_to_vertex_right, vertex_to_id_left, vertex_to_id_right)
  end
end

struct BipartiteMatchingSolution{T}
  instance::BipartiteMatchingInstance{T}
  solution::Vector{Edge{T}}
  value::Float64
end

function matching_hungarian(i::ipartiteMatchingInstance{T}) where T
  # No implementation of the Hungarian algorithm for matchings: reuse Munkres.jl.
  reward_matrix = zeros(i.n_left, i.n_right)
  for (k, v) in i.reward
    # idx1: always left. idx2: always right.
    if src(k) in keys(i.vertex_to_id_left)
      idx1 = i.vertex_to_id_left[src(k)]
      idx2 = i.vertex_to_id_right[dst(k)]
    else
      idx1 = i.vertex_to_id_left[dst(k)]
      idx2 = i.vertex_to_id_right[src(k)]
    end

    reward_matrix[idx1, idx2] = v
  end

  # Make a cost matrix, to match the objective function of the underlying solver.
  cost_matrix = maximum(reward_matrix) .- reward_matrix

  # Solve the corresponding matching problem. Discard the returned cost, as it makes no sense with the previous solution.
  sol, _ = hungarian(cost_matrix)

  # Return a list of edges in the graph, restoring the numbers of each vertex.
  solution = Edge{T}[]
  value = 0.0
  for (idx1, idx2) in enumerate(sol)
    v1 = i.id_to_vertex_left[idx1]
    v2 = i.id_to_vertex_right[idx2]
    e = Edge(v1, v2)

    push!(e)
    value += i.reward[e]
  end

  return BipartiteMatchingSolution(i, solution, value)
end
