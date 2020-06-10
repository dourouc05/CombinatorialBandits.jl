# ESCB-2, "efficient sampling for combinatorial bandits".
# Based on https://arxiv.org/abs/1502.03475.

abstract type ESCB2OptimisationAlgorithm end
function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, ::ESCB2OptimisationAlgorithm,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64}, round::Int) where T end

mutable struct ESCB2 <: Policy
  algo::ESCB2OptimisationAlgorithm
end

mutable struct ESCB2Details <: PolicyDetails
  n_iterations::Int
  best_objective::Float64
  solver_time::Float64

  function ESCB2Details()
    return new(0, 0.0, 0.0)
  end
end

function _escb2_confidence_values(state::State{T}) where T
  t = state.arm_counts
  w = state.arm_average_reward
  n = state.round

  d = length(t) # n_arms
  fn = log(n) + 4 * d * log(log(n))
  fn = max(0, fn) # f is negative for some values of d and small values of n (sometimes as small as 2 or 3).

  return Dict(k => (fn / 2) / v for (k, v) in t)
end

# TODO: shouldn't it go to a more generic function? Many algorithm work with indices, so maybe there is something to do... but only for black-box algorithms?
function escb2_index(state::State{T}, solution::Vector{T}) where T
  w = state.arm_average_reward
  s2 = _escb2_confidence_values(state)
  return escb2_index(w, s2, solution)
end
function escb2_index(linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64}, solution::Vector{T}) where T
  if length(solution) == 0
    return 0.0
  end

  avg_reward = sum(linear[arm] for arm in keys(linear) if arm in solution)
  confidence = sum(sqrtlinear[arm] for arm in keys(sqrtlinear) if arm in solution)
  return avg_reward + sqrt(confidence)
end

function choose_action(instance::CombinatorialInstance{T}, policy::ESCB2, state::State{T}; with_trace::Bool=false) where T
  # For initialisation, if some arms have never been tried, force them to be tried.
  # Without this, the standard deviation term cannot exist.
  if any(v == 0 for v in values(state.arm_counts))
    weights = Dict(arm => iszero(state.arm_counts[arm]) ? 1.0 : 0.0 for arm in keys(state.arm_counts))

    t0 = time_ns()
    sol = solve_linear(instance, weights)
    t1 = time_ns()

    if with_trace
      run_details = ESCB2Details()
      run_details.n_iterations = 1
      run_details.solver_time = (t1 - t0) / 1_000_000_000
      return sol, run_details
    else
      return sol
    end
  end

  # If all arms have been tried, can compute the two parts of the objective function: the average reward and a standard deviation.
  w = state.arm_average_reward
  s2 = _escb2_confidence_values(state)

  # Solve the maximisation w^T x + \sqrt{s2^T x} with the required algorithm.
  return optimise_linear_sqrtlinear(instance, policy.algo, w, s2, state.round, with_trace=with_trace)
end
