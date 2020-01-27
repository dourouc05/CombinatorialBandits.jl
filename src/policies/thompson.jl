# Thompson sampling.
# Based on https://arxiv.org/abs/1111.1797.

mutable struct ThompsonSampling <: Policy end

struct ThompsonSamplingDetails <: PolicyDetails
  solver_time::Float64
end

function choose_action(instance::CombinatorialInstance{T}, policy::ThompsonSampling, state::State{T}; with_trace::Bool=false) where T
  # Determine the weights for each arm and use them to solve the combinatorial problem.
  # println("Counts: $(state.arm_counts); rewards: $(state.arm_average_reward)")
  weights = Dict(arm => rand(Beta(state.arm_counts[arm] * state.arm_average_reward[arm] + 1,
                                  state.arm_counts[arm] * (1 - state.arm_average_reward[arm]) + 1)) for arm in keys(state.arm_counts))

  t0 = time_ns()
  sol = solve_linear(instance, weights)
  t1 = time_ns()

  if with_trace
    return sol, ThompsonSamplingDetails((t1 - t0) / 1_000_000_000)
  else
    return sol
  end
end
