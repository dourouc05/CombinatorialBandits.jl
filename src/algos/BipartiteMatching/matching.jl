struct BipartiteMatchingInstance{T} <: CombinatorialInstance
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

    return new{T}(copy(graph), copy(reward), bmap, n_left, n_right, vertex_left, vertex_right)
  end

  function BipartiteMatchingInstance(i::BipartiteMatchingInstance{T}, reward::Dict{Edge{T}, Float64}) where T
    # Only for private use. Change the rewards of a given instance.
    return new{T}(i.graph, reward, i.partition, i.n_left, i.n_right, i.vertex_left, i.vertex_right)
  end
end

struct BipartiteMatchingSolution{T} <: CombinatorialSolution
  instance::BipartiteMatchingInstance{T}
  solution::Vector{Edge{T}}
  value::Float64
end
