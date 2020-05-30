using LightGraphs
using DataStructures

struct SpanningTreeInstance{T} <: CombinatorialInstance
  graph::AbstractGraph{T}
  rewards::Dict{Edge{T}, Float64}
end

graph(i::SpanningTreeInstance{T}) where T = i.graph
# rewards(i::SpanningTreeInstance{T}) where T = i.rewards
function reward(i::SpanningTreeInstance{T}, e::Edge{T}) where T
  if e in keys(i.rewards)
    return i.rewards[e]
  end
  return i.rewards[reverse(e)]
end

struct SpanningTreeSolution{T} <: CombinatorialSolution
  instance::SpanningTreeInstance{T}
  tree::Vector{Edge{T}}
end
