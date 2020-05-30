mutable struct MSetAlgosSolver <: MSetSolver
  m # Int
  n_arms # Int

  function MSetAlgosSolver()
    return new(nothing, nothing)
  end
end

copy(::MSetAlgosSolver) = MSetAlgosSolver()

function build!(solver::MSetAlgosSolver, m::Int, n_arms::Int)
  solver.m = m
  solver.n_arms = n_arms
end

has_lp_formulation(::MSetAlgosSolver) = false
approximation_ratio(::MSetAlgosSolver) = 1.0
approximation_term(::MSetAlgosSolver) = 0.0
approximation_ratio_budgeted(::MSetAlgosSolver) = 1.0 # TODO: redirect to the package.
approximation_term_budgeted(::MSetAlgosSolver) = 0.0
supports_solve_budgeted_linear(::MSetAlgosSolver) = true
supports_solve_all_budgeted_linear(::MSetAlgosSolver) = true

function _rewards_weights_dict_to_vectors(rewards::Dict{Int, Float64}, weights::Dict{Int, Int})
  objects = vcat(collect(keys(rewards)), collect(keys(weights)))
  unique!(objects)
  sort!(objects)

  rewards_vector = [rewards[o] for o in objects]
  weights_vector = [weights[o] for o in objects]
  return rewards_vector, weights_vector
end

function solve_linear(solver::MSetAlgosSolver, weights::Dict{Int, Float64})
  objects = collect(keys(weights))
  sort!(objects)
  weights_vector = [weights[o] for o in objects]

  return msets_greedy(MSetInstance(weights_vector, solver.m)).items
end

function solve_budgeted_linear(solver::MSetAlgosSolver, rewards::Dict{Int, Float64}, weights::Dict{Int, Int}, budget::Int)
  rewards_vector, weights_vector = _rewards_weights_dict_to_vectors(rewards, weights)
  i = BudgetedMSetInstance(rewards_vector, weights_vector, solver.m, budget=budget)
  return budgeted_msets_dp(i).items
end

function solve_all_budgeted_linear(solver::MSetAlgosSolver, rewards::Dict{Int, Float64}, weights::Dict{Int, Int}, max_budget::Int)
  rewards_vector, weights_vector = _rewards_weights_dict_to_vectors(rewards, weights)
  i = BudgetedMSetInstance(rewards_vector, weights_vector, solver.m, budget=max_budget)
  s = budgeted_msets_dp(i)
  return items_all_budgets(s, max_budget)
end
