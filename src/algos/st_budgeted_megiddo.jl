function st_prim_budgeted_lagrangian_megiddo(i::SpanningTreeInstance{T}, budget::Int) where T
  # Initial set of values for λ.
  λlow = 0
  λhigh = Inf

  # TODO: add a fake edge in the graph for the remaining term.

  # TODO: maybe use https://pdfs.semanticscholar.org/a11c/7d19cc2f06fc4d1f8d5b8dcbf8e87e372147.pdf for implementation?
  # Or other references from https://www2.cs.duke.edu/courses/spring07/cps296.2/scribe_notes/p412-agarwal.pdf.

  # Start Prim's algorithm.
  remaining_edges = PriorityQueue{Edge{T}, Float64}() # Easy retrieval of minimum-cost edge.
  node_done = falses(nv(graph(i)))
  node_done[first(vertices(graph(i)))] = true

  # Helper methods.
  edges_around(g, v) = (edgetype(g)(v, x) for x in neighbors(g, v))

  # Initialise with the source node (arbitrarily, the first one).
  first_node = first(vertices(graph(i)))
  for e in edges_around(graph(i), first_node)
    enqueue!(remaining_edges, e => cost(i, e))
  end

  # Build the spanning tree.
  solution = Edge{T}[]
  current_node = first_node
  while length(solution) < nv(graph(i)) - 1 # While there is still an edge to add...
    # Find an admissible edge, starting with those of minimum cost.
    e = nothing
    while true
      e = dequeue!(remaining_edges)
      if (node_done[src(e)] && ! node_done[dst(e)]) || (! node_done[src(e)] && node_done[dst(e)])
        break
      end
    end

    # Update the current node.
    if src(e) == current_node
      current_node = dst(e)
    else
      current_node = src(e)
    end

    # Use this edge in the solution.
    push!(solution, e)
    node_done[current_node] = true

    # Prepare to use the neighbours of this new node.
    for e in edges_around(graph(i), current_node)
      enqueue!(remaining_edges, e => cost(i, e))
    end
  end

  return SpanningTreeSolution(i, solution)
end
