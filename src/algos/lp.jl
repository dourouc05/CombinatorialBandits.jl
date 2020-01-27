using LightGraphs

import LightGraphs: src, dst

struct ElementaryPathInstance{T}
  graph::AbstractGraph{T}
  costs::Dict{Edge{T}, Float64}
  src::T
  dst::T
end

graph(i::ElementaryPathInstance{T}) where T = i.graph
costs(i::ElementaryPathInstance{T}) where T = i.costs
src(i::ElementaryPathInstance{T}) where T = i.src
dst(i::ElementaryPathInstance{T}) where T = i.dst

dimension(i::ElementaryPathInstance{T}) where T = ne(graph(i))
cost(i::ElementaryPathInstance{T}, u::T, v::T) where T = costs(i)[Edge(u, v)]

struct ElementaryPathSolution{T}
  instance::ElementaryPathInstance{T}
  path::Vector{Edge{T}}
  states::Dict{T, Float64}
  solutions::Dict{T, Vector{Edge{T}}}
end

function lp_dp(i::ElementaryPathInstance{T}) where T # I.e. Bellman-Ford algorithm. Assumption: no positive-cost cycle in the graph.
  V = Dict{T, Float64}()
  S = Dict{T, Vector{Edge{T}}}()

  # Initialise.
  for v in vertices(i.graph)
    V[v] = -Inf
    S[v] = Edge{T}[]
  end
  V[i.src] = 0.0

  # Dynamic part.
  for _ in 1:ne(i.graph)
    changes = false # Stop the algorithm as soon as it no more makes progress.

    for e in edges(i.graph)
      u, v = src(e), dst(e)
      w = cost(i, u, v)

      # If using the solution to the currently explored subproblem would
      # lead to a cycle, skip it.
      if any(src(e) == v for e in S[u])
        continue
      end

      # Compute the maximum: is passing through w advantageous?
      if V[u] + w > V[v]
        V[v] = V[u] + w
        S[v] = vcat(S[u], Edge(u, v))

        changes = true
      end
    end

    if ! changes
      break
    end
  end

  # Checking existence of negative-cost cycle.
  for e in edges(graph(i))
    u, v = src(e), dst(e)
    w = cost(i, u, v)

    if V[u] + w > V[v]
      @warn("The graph contains a positive-cost cycle around edge $(u) -> $(v).")
    end
  end

  return ElementaryPathSolution(i, S[i.dst], V, S)
end
