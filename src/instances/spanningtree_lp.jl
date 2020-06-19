# Unlike other LP implementations, this one is not totally unimodular, and
# thus requires an ILP solver.
mutable struct SpanningTreeLPSolver <: SpanningTreeSolver
  graph::SimpleDiGraph
  weights_matrix::Matrix{Float64}

  solver

  # Optimisation model (reused by solve_linear).
  model # ::Model
  x # ::Vector{Variable}: index is edge
  flow # ::Vector{Variable}: index is edge

  function SpanningTreeLPSolver(solver)
    return new(DiGraph(0), zeros(0, 0), solver, nothing, nothing)
  end
end

has_lp_formulation(::SpanningTreeLPSolver) = true
supports_solve_budgeted_linear(::SpanningTreeLPSolver) = false
supports_solve_all_budgeted_linear(::SpanningTreeLPSolver) = false

function build!(solver::SpanningTreeLPSolver, graph::SimpleGraph)
  # Input graph supposed to be undirected.
  n = nv(graph)
  solver.graph = DiGraph(n)
  solver.weights_matrix = zeros(n, n)

  # Build the equivalent directed graph.
  for e in edges(graph)
    add_edge!(solver.graph, src(e), dst(e))
    add_edge!(solver.graph, dst(e), src(e))
  end

  # Helper methods.
  inedges(g, v) = (edgetype(g)(x, v) for x in inneighbors(g, v))
  outedges(g, v) = (edgetype(g)(v, x) for x in outneighbors(g, v))

  # Build the optimisation model behind solve_linear.
  # Based on Magnanti, T.L.; Wolsey, L. Optimal Trees, section Flow formulation (p. 38).
  solver.model = Model(solver.solver)
  solver.x = @variable(solver.model, [e in edges(solver.graph)], binary=true)
  solver.flow = @variable(solver.model, [e in edges(solver.graph)], lower_bound=0)

  for e in edges(solver.graph)
    set_name(solver.x[e], "x_$(src(e))_$(dst(e))")
    set_name(solver.flow[e], "flow_$(src(e))_$(dst(e))")
  end

  # Choose a source arbitrarily.
  source = first(vertices(solver.graph))
  other_nodes = filter(v -> v != source, vertices(solver.graph))

  @constraint(solver.model, sum(solver.flow[e] for e in inedges(solver.graph, source)) == 0)
  @constraint(solver.model, sum(solver.flow[e] for e in outedges(solver.graph, source)) == n - 1)
  for v in other_nodes
    @constraint(solver.model, sum(solver.flow[e] for e in inedges(solver.graph, v)) - sum(solver.flow[e] for e in outedges(solver.graph, v)) == 1)
  end

  @constraint(solver.model, sum(solver.x) == n - 1)
  for e in edges(solver.graph)
    @constraint(solver.model, solver.flow[e] <= (n - 1) * solver.x[e])
  end

  for e in edges(graph)
    @constraint(solver.model, solver.x[e] + solver.x[reverse(e)] <= 1)
  end
end

function get_lp_formulation(solver::SpanningTreeLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  obj = sum(reward[i, j] * (solver.x[Edge(i, j)] + solver.x[Edge(j, i)]) for (i, j) in keys(reward))
  vars_forw = Dict{Tuple{Int, Int}, JuMP.VariableRef}((i, j) => solver.x[Edge(i, j)] for (i, j) in keys(reward))
  vars_back = Dict{Tuple{Int, Int}, JuMP.VariableRef}((j, i) => solver.x[Edge(j, i)] for (i, j) in keys(reward))
  vars = merge(vars_forw, vars_back)

  return solver.model, obj, vars
end

function solve_linear(solver::SpanningTreeLPSolver, reward::Dict{Tuple{Int, Int}, Float64})
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Maybe solve_budgeted_linear has already been called on this model.
  if :SpanningTreeLP in keys(m.ext)
    budget_constraint = m.ext[:SpanningTreeLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, 0)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end

  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5 || value(vars[j, i]) > 0.5]
end

function solve_budgeted_linear(solver::SpanningTreeLPSolver,
                               reward::Dict{Tuple{Int, Int}, Float64},
                               weight::Dict{Tuple{Int, Int}, T},
                               budget::Int) where {T<:Number} # Handle both Int and Float64
  m, obj, vars = get_lp_formulation(solver, reward)
  @objective(m, Max, obj)

  # Add the budget constraint (or change the existing constraint).
  if :SpanningTreeLP in keys(m.ext)
    budget_constraint = m.ext[:SpanningTreeLP][:budget_constraint]
    set_normalized_rhs(budget_constraint, budget)
  else
    budget_constraint = @constraint(m, sum(weight[i] * vars[i] for i in keys(reward)) >= budget)
    m.ext[:SpanningTreeLP] = Dict(:budget_constraint => budget_constraint)
  end

  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    return Tuple{Int, Int}[]
  end
  return Tuple{Int, Int}[(i, j) for (i, j) in keys(reward) if value(vars[i, j]) > 0.5 || value(vars[j, i]) > 0.5]
end
