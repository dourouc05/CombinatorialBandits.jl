struct ElementaryPathInstance{T} <: CombinatorialInstance
  graph::AbstractGraph{T}
  costs::Dict{Edge{T}, Float64}
  src::T
  dst::T
end

graph(i::ElementaryPathInstance{T}) where T = i.graph
src(i::ElementaryPathInstance{T}) where T = i.src
dst(i::ElementaryPathInstance{T}) where T = i.dst

# dimension(i::ElementaryPathInstance{T}) where T = ne(graph(i))
cost(i::ElementaryPathInstance{T}, u::T, v::T) where T = i.costs[Edge(u, v)]

struct ElementaryPathSolution{T} <: CombinatorialInstance
  instance::ElementaryPathInstance{T}
  path::Vector{Edge{T}}
  states::Dict{T, Float64}
  solutions::Dict{T, Vector{Edge{T}}}
end
