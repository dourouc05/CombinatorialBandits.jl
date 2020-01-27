mutable struct SpanningTreeAlgosSolver <: SpanningTreeSolver
  graph::SimpleGraph

  function SpanningTreeAlgosSolver()
    return new(Graph(0))
  end
end

function build!(solver::SpanningTreeAlgosSolver, graph::SimpleGraph)
  solver.graph = graph
end

function solve_linear(solver::SpanningTreeAlgosSolver, reward::Dict{Tuple{Int, Int}, Float64})
  reward_dict = Dict(Edge(k...) => v for (k, v) in reward)

  i = SpanningTreeInstance(solver.graph, reward_dict)
  s = st_prim(i).tree

  return _mst_solution_normalise(reward, Tuple{Int, Int}[(src(e), dst(e)) for e in s])
end

has_lp_formulation(::SpanningTreeAlgosSolver) = false

function solve_budgeted_linear(solver::SpanningTreeAlgosSolver, reward::Dict{Tuple{Int, Int}, Float64}, weight::Dict{Tuple{Int, Int}, Int}, budget::Int)
  reward_dict = Dict(Edge(k...) => v for (k, v) in reward)
  weight_dict = Dict(Edge(k...) => v for (k, v) in weight)

  i = BudgetedSpanningTreeInstance(solver.graph, reward_dict, weight_dict, budget)
  s = st_prim_budgeted_lagrangian_refinement(i).tree

  return _mst_solution_normalise(reward, Tuple{Int, Int}[(src(e), dst(e)) for e in s])
end
