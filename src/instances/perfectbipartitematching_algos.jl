mutable struct PerfectBipartiteMatchingAlgosSolver <: PerfectBipartiteMatchingSolver
  # All other instances only work with complete bipartite graphs, do the same here.
  n_arms::Int

  function PerfectBipartiteMatchingAlgosSolver()
    new(-1)
  end
end

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

has_lp_formulation(::PerfectBipartiteMatchingAlgosSolver) = false
