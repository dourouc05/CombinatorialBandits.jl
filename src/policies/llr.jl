# LLR, "learning with linear rewards".
# Based on https://arxiv.org/abs/1011.4748.

mutable struct LLR <: Policy 
  α::Float64
end

LLR() = LLR(1.0)

struct LLRDetails <: PolicyDetails
  solver_time::Float64
end

function choose_action(instance::CombinatorialInstance{T}, algo::LLR, state::State{T}; with_trace::Bool=false) where T
  if any(v == 0 for v in values(state.arm_counts))
    # Initialisation step: for each random variable (i.e. reward source), try arms, preferring those that have never been tested.
    weights = Dict(arm => (state.arm_counts[arm] == 0.0) ? 1.0 : 0.0 for arm in keys(state.arm_counts))
  else
    # Determine the weights for each arm and use them to solve the combinatorial problem.
    m = maximum_solution_length(instance)
    numerator = algo.α * sqrt((m + 1) * log(state.round))
    weights = Dict(arm => state.arm_average_reward[arm] + numerator * sqrt(1.0 / state.arm_counts[arm]) for arm in keys(state.arm_counts))
  end

  t0 = time_ns()
  sol = solve_linear(instance, weights)
  t1 = time_ns()

  if with_trace
    return sol, LLRDetails((t1 - t0) / 1_000_000_000)
  else
    return sol
  end
end
