mutable struct ElementaryPathLPSolver <: ElementaryPathSolver
  graph::SimpleDiGraph
  source::Int
  destination::Int

  solver

  # Optimisation model (reused by solve_linear).
  model # ::Model
  x # ::Vector{Variable}: index is edge

  function ElementaryPathLPSolver(solver)
    return new(DiGraph(0), -1, -1, solver, nothing, nothing)
  end
end

function _elementary_path_lazy_callback(solver::ElementaryPathLPSolver, cb_data)
  # Based on the hypothesis that the strengthening constraints have been added beforehand!
  # I.e. at most one edge incoming and at most one outgoing (except for source/destination).
  x = Dict(e => callback_value(cb_data, solver.x[e]) for e in edges(solver.graph))

  # Helper methods.
  inedges(g, v) = (edgetype(g)(x, v) for x in inneighbors(g, v))
  outedges(g, v) = (edgetype(g)(v, x) for x in outneighbors(g, v))

  # Find a subtour.
  current_node = solver.source # TODO: factor this loop out? Also useful for sorting edges.
  while current_node != solver.destination
    # Only one outgoing edge from the current node, as per the constraints.
    for e in outedges(solver.graph, current_node)
      if x[e] >= .5
        x[e] = 0.0
        current_node = dst(e)
        break
      end
    end
  end

  # Find the edges that remain after the path is removed.
  lhs_edges = Edge[]
  for e in edges(solver.graph)
    # Don't consider edges adjacent to the source or the destination
    if src(e) == solver.source || dst(e) == solver.destination
      continue
    end

    if x[e] >= .5
      push!(lhs_edges, e)
    end
  end

  # Nothing left? No subtour!
  if length(lhs_edges) == 0
    return
  end

  # Add lazy constraints (subtour elimination). Separate all subtours to generate constraints
  # as tight as possible.
  while length(lhs_edges) > 0
    # Find one subtour among these edges.
    con_edges = Edge[lhs_edges[1]]
    cur_node = dst(lhs_edges[1])
    while cur_node != src(lhs_edges[1])
      for (i, e) in enumerate(lhs_edges)
        if src(e) == cur_node
          cur_node = dst(e)
          push!(con_edges, e)
          deleteat!(lhs_edges, i)
        end
      end
    end

    # Build the corresponding constraint.
    con = @build_constraint(sum(x[e] for e in con_edges) <= length(con_edges) - 1)
    MOI.submit(solver.model, MOI.LazyConstraint(cb_data), con)
  end
end

function build!(solver::ElementaryPathLPSolver, graph::SimpleDiGraph, source::Int, destination::Int)
  n = nv(graph)
  solver.graph = graph
  solver.source = source
  solver.destination = destination

  # Helper methods.
  inedges(g, v) = (edgetype(g)(x, v) for x in inneighbors(g, v))
  outedges(g, v) = (edgetype(g)(v, x) for x in outneighbors(g, v))

  other_nodes = collect(filter(vertices(solver.graph)) do v; v != source && v != destination; end)

  # Build the optimisation model behind solve_linear.
  solver.model = Model(solver.solver)
  solver.x = @variable(solver.model, [e in edges(solver.graph)], binary=true)
  # Normally, with this formulation, explicitly having binary variables is not necessary,
  # but this is only valid if minimising a linear function. In all other cases, there is
  # no guarantee to have an integer-feasible solution.

  for e in edges(solver.graph)
    set_name(solver.x[e], "x_$(src(e))_$(dst(e))")
  end

  function edge_incidence(v)
    if length(inedges(solver.graph, v)) > 0
      ins = sum(solver.x[e] for e in inedges(solver.graph, v))
    else
      ins = 0.0
    end

    if length(outedges(solver.graph, v)) > 0
      outs = sum(solver.x[e] for e in outedges(solver.graph, v))
    else
      outs = 0.0
    end

    return ins - outs
  end

  @constraint(solver.model, edge_incidence(source) == -1)
  @constraint(solver.model, edge_incidence(destination) == 1)
  @constraint(solver.model, [v in other_nodes], edge_incidence(v) == 0)

  # # Eliminate a large number of subtours.
  @constraint(solver.model, sum(solver.x[e] for e in inedges(solver.graph, source)) == 0)
  @constraint(solver.model, sum(solver.x[e] for e in outedges(solver.graph, source)) == 1)
  @constraint(solver.model, sum(solver.x[e] for e in inedges(solver.graph, destination)) == 1)
  @constraint(solver.model, sum(solver.x[e] for e in outedges(solver.graph, destination)) == 0)
  for v in other_nodes
    @constraint(solver.model, sum(solver.x[e] for e in inedges(solver.graph, v)) <= 1)
    @constraint(solver.model, sum(solver.x[e] for e in outedges(solver.graph, v)) <= 1)
  end

  # Set the lazy-constraint callback to ensure the solution is always an elementary path.
  set_parameter(solver.model, "LazyConstraints", 1) # TODO: only works with Gurobi.
  MOI.set(solver.model, MOI.LazyConstraintCallback(), cb_data -> _elementary_path_lazy_callback(solver, cb_data))
end

has_lp_formulation(::ElementaryPathLPSolver) = true

function get_lp_formulation(solver::ElementaryPathLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  return solver.model,
    sum(reward[(i, j)] * solver.x[Edge(i, j)] for (i, j) in keys(reward)),
    Dict{Tuple{Int, Int}, JuMP.VariableRef}((i, j) => solver.x[Edge(i, j)] for (i, j) in keys(reward))
end

function _sort_path(path::Vector{Tuple{Int, Int}}, source::Int, destination::Int, n_vertices::Int)
  sorted = Tuple{Int, Int}[]
  current_node = source
  edges = copy(path)

  i = 0
  while length(edges) > 0
    # Find the corresponding edge (or the first one that matches).
    for i in 1:length(edges)
      e = edges[i]
      if e[1] == current_node
        push!(sorted, e)
        current_node = e[2]
        deleteat!(edges, i)
        break
      end
    end

    if current_node == destination
      break
    end

    # Safety: ensure this loop does not make too many iterations.
    i += 1
    if i > n_vertices
      error("Assertion failed: infinite loop when sorting the edges of the path $path")
    end
  end

  if length(edges) > 0
    error("Edges remaining after sorting the edges: the solution is likely to contain at least a subtour, $edges")
  end

  return sorted
end

function solve_linear(solver::ElementaryPathLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end
  sol = Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5]
  return _sort_path(sol, solver.source, solver.destination, nv(solver.graph))
end

function solve_budgeted_linear(solver::ElementaryPathLPSolver,
                               reward::Dict{Tuple{Int, Int}, Float64},
                               weight::Dict{Tuple{Int, Int}, T},
                               budget::Int) where {T<:Number} # Handle both Int and Float64
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  if :ElementaryPathLP in keys(m.ext)
    budget_constraint = m.ext[:ElementaryPathLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weight[i] * vars[i] for i in keys(reward)) >= budget)
    m.ext[:ElementaryPathLP] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end

  sol = Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5]
  return _sort_path(sol, solver.source, solver.destination, nv(solver.graph))
end
