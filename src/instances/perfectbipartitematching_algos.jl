mutable struct PerfectBipartiteMatchingAlgosSolver <: PerfectBipartiteMatchingSolver
  # All other instances only work with complete bipartite graphs, do the same here.
  n_arms::Int

  function PerfectBipartiteMatchingAlgosSolver()
    new(-1)
  end
end

supports_solve_budgeted_linear(::PerfectBipartiteMatchingAlgosSolver) = true
supports_solve_all_budgeted_linear(::PerfectBipartiteMatchingAlgosSolver) = true

function build!(solver::PerfectBipartiteMatchingAlgosSolver, n_arms::Int)
  solver.n_arms = n_arms
  nothing
end

function solve_linear(solver::PerfectBipartiteMatchingAlgosSolver, rewards::Dict{Tuple{Int, Int}, Float64})
  g = complete_bipartite_graph(solver.n_arms, solver.n_arms)
  r = Dict{Edge{Int}, Float64}(Edge(k[1], solver.n_arms + k[2]) => v for (k, v) in rewards)
  i = BipartiteMatchingInstance(g, r)

  s = matching_hungarian(i).solution

  return [(src(e), dst(e) - solver.n_arms) for e in s]
end

function solve_budgeted_linear(solver::PerfectBipartiteMatchingAlgosSolver,
                               rewards::Dict{Tuple{Int, Int}, Float64},
                               weights::Dict{Tuple{Int, Int}, Int},
                               budget::Int)
  g = complete_bipartite_graph(solver.n_arms, solver.n_arms)
  r = Dict{Edge{Int}, Float64}(Edge(k[1], solver.n_arms + k[2]) => v for (k, v) in rewards)
  w = Dict{Edge{Int}, Int}(Edge(k[1], solver.n_arms + k[2]) => v for (k, v) in weights)
  i = BudgetedBipartiteMatchingInstance(g, r, w, budget)

  s = matching_hungarian_budgeted_lagrangian_approx_half(i).solution

  return [(src(e), dst(e) - solver.n_arms) for e in s]
end

function solve_all_budgeted_linear(solver::PerfectBipartiteMatchingAlgosSolver,
                                   rewards::Dict{Tuple{Int, Int}, Float64},
                                   weights::Dict{Tuple{Int, Int}, Int},
                                   max_budget::Int)
  g = complete_bipartite_graph(solver.n_arms, solver.n_arms)
  r = Dict{Edge{Int}, Float64}(Edge(k[1], solver.n_arms + k[2]) => v for (k, v) in rewards)
  w = Dict{Edge{Int}, Int}(Edge(k[1], solver.n_arms + k[2]) => v for (k, v) in weights)
  i = BudgetedBipartiteMatchingInstance(g, r, w, max_budget)

  s = matching_dp_budgeted(i).solution

  sol = Dict{Int, Vector{Edge{Int}}}()
  vl = last(instance.matching.vertex_left)
  vr = last(instance.matching.vertex_right)
  for budget in 0:max_budget
    sol[budget] = s.solutions[vl, vr, 0, budget]
  end
  return sol
end

has_lp_formulation(::PerfectBipartiteMatchingAlgosSolver) = false
approximation_ratio(::PerfectBipartiteMatchingAlgosSolver) = 1.0
approximation_term(::PerfectBipartiteMatchingAlgosSolver) = 0.0
approximation_ratio_budgeted(::PerfectBipartiteMatchingAlgosSolver) = 0.5
approximation_term_budgeted(::PerfectBipartiteMatchingAlgosSolver) = 0.0
