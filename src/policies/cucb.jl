# CUCB, "combinatorial upper confidence bound".
# Based on http://proceedings.mlr.press/v28/chen13a.html.

mutable struct CUCB <: Policy 
  α::Float64
end

CUCB() = CUCB(sqrt(0.5))

struct CUCBDetails <: PolicyDetails
  solver_time::Float64
end

function choose_action(instance::CombinatorialInstance{T}, algo::CUCB, state::State{T}; with_trace::Bool=false) where T
  if any(v == 0 for v in values(state.arm_counts))
    # There is at least one arm that has never been tried: maximise the arms that have never been tested.
    weights = Dict(arm => (state.arm_counts[arm] == 0.0) ? 1.0 : 0.0 for arm in keys(state.arm_counts))
  else
    # All arms have been seen, thus this formula makes sense (no zero-valued arm count).
    numerator = algo.α * sqrt(log(state.round))
    weights = Dict(arm => state.arm_average_reward[arm] + numerator * sqrt(1.0 / state.arm_counts[arm]) for arm in keys(state.arm_counts))
  end

  t0 = time_ns()
  sol = solve_linear(instance, weights)
  t1 = time_ns()

  if with_trace
    return sol, CUCBDetails((t1 - t0) / 1_000_000_000)
  else
    return sol
  end
end
