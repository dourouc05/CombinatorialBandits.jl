solve(i::ElementaryPathInstance{T}, ::BellmanFordAlgorithm; kwargs...) where T = lp_dp(i; kwargs...)

function lp_dp(i::ElementaryPathInstance{T}) where T # I.e. Bellman-Ford algorithm. Assumption: no positive-cost cycle in the graph.
  V = Dict{T, Float64}()
  S = Dict{T, Vector{Edge{T}}}()

  # Initialise.
  for v in vertices(graph(i))
    V[v] = -Inf
    S[v] = Edge{T}[]
  end
  V[src(i)] = 0.0

  # Dynamic part.
  for _ in 1:ne(graph(i))
    changes = false # Stop the algorithm as soon as it no more makes progress.

    for e in edges(graph(i))
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

  return ElementaryPathSolution(i, S[dst(i)], V, S)
end