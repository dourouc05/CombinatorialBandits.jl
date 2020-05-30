# Thompson sampling.
# Based on https://arxiv.org/abs/1111.1797.

# Argument is the type of conjugate distribution. 
# Assumption if needed: unit variance. (Only useful for normal distribution/priors, for now.)
mutable struct ThompsonSampling{D <: Distribution} <: Policy end

ThompsonSampling() = ThompsonSampling{Normal}() # Default prior distribution: normal.

struct ThompsonSamplingDetails <: PolicyDetails
  solver_time::Float64
end

# Beta priors: assume initially α = β = 1 (i.e. the prior state is to have observed one failure and one success 
# for each arm). 
function _ts_sample(::ThompsonSampling{Beta}, state::State{T}) where T
  return Dict(arm => rand(Beta(state.arm_counts[arm] * state.arm_average_reward[arm] + 1,
                               state.arm_counts[arm] * (1 - state.arm_average_reward[arm]) + 1)) 
                     for arm in keys(state.arm_counts))
end

# Normal priors: assume that the arms have a normal distribution with unknown average and known unit variance. 
# Assume that the initial average is zero. This simplifies the formulae a lot!
function _ts_sample(::ThompsonSampling{Normal}, state::State{T}) where T
  n(arm) = state.arm_counts[arm]
  tr(arm) = state.arm_reward[arm]
  return Dict(arm => rand(Normal(tr(arm) / (1 + n(arm)), 1 / (1 + n(arm)))) for arm in keys(state.arm_counts))
end

function choose_action(instance::CombinatorialInstance{T}, algo::ThompsonSampling{D}, state::State{T}; with_trace::Bool=false) where {T, D}
  weights = if any(v == 0 for v in values(state.arm_counts))
    # Initialisation step: for each random variable (i.e. reward source), try arms, preferring those that have never been tested.
    Dict(arm => (state.arm_counts[arm] == 0.0) ? 1.0 : 0.0 for arm in keys(state.arm_counts))
  else
    # Normal step: just sample the prior distribution.
    _ts_sample(algo, state)
  end

  # Avoid having all weights negative, this will yield a zero solution.
  if all(values(weights) .<= 0)
    min_val = maximum(values(weights))
    for k in keys(weights)
      weights[k] -= 2 * min_val
    end
  end

  # Actually find the solution to play.
  t0 = time_ns()
  sol = solve_linear(instance, weights)
  t1 = time_ns()

  if with_trace
    return sol, ThompsonSamplingDetails((t1 - t0) / 1_000_000_000)
  else
    return sol
  end
end
