mutable struct ElementaryPathAlgosSolver <: ElementaryPathSolver
  graph::SimpleDiGraph
  source::Int
  destination::Int

  function ElementaryPathAlgosSolver()
    return new(DiGraph(0), -1, -1)
  end
end

function build!(solver::ElementaryPathAlgosSolver, graph::SimpleDiGraph, source::Int, destination::Int)
  solver.graph = graph
  solver.source = source
  solver.destination = destination
end

has_lp_formulation(::ElementaryPathAlgosSolver) = false

_tuple_dict_to_edge_dict(d::Dict{Tuple{Int, Int}, T}) where T =
  Dict(Edge(k...) => v for (k, v) in d)

function solve_linear(solver::ElementaryPathAlgosSolver, reward::Dict{Tuple{Int, Int}, Float64})
  i = ElementaryPathInstance(solver.graph, _tuple_dict_to_edge_dict(reward),
    solver.source, solver.destination)
  s = lp_dp(i).path
  return [(src(e), dst(e)) for e in s]
end

function solve_budgeted_linear(solver::ElementaryPathAlgosSolver,
                               reward::Dict{Tuple{Int, Int}, Float64},
                               weights::Dict{Tuple{Int, Int}, Int},
                               budget::Int)
  i = BudgetedElementaryPathInstance(solver.graph,
    _tuple_dict_to_edge_dict(reward), _tuple_dict_to_edge_dict(weights),
    solver.source, solver.destination, budget=budget)
  s = budgeted_lp_dp(i).path
  return [(src(e), dst(e)) for e in s]
end

function solve_all_budgeted_linear(solver::ElementaryPathAlgosSolver,
                                   reward::Dict{Tuple{Int, Int}, Float64},
                                   weights::Dict{Tuple{Int, Int}, Int},
                                   max_budget::Int)
  i = BudgetedElementaryPathInstance(solver.graph,
    _tuple_dict_to_edge_dict(reward), _tuple_dict_to_edge_dict(weights),
    solver.source, solver.destination, budget=max_budget)
  return paths_all_budgets_as_tuples(budgeted_lp_dp(i), max_budget)
end
