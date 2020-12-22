# ESCB-2, "efficient sampling for combinatorial bandits".
# Based on https://arxiv.org/abs/1502.03475.

abstract type ESCB2OptimisationAlgorithm end

"""
  function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T},
                                      algo::ESCB2OptimisationAlgorithm,
                                      linear::Dict{T, Float64},
                                      sqrtlinear::Dict{T, Float64},
                                      sqrtlinear_weight::Float64, 
                                      bandit_round::Int;
                                      with_trace::Bool=false) where T

Optimise ESCB2's objective function, with a linear term (with coefficients in `linear`)
and a square root (with coefficients in `sqrtlinear`): 

  max linear^T x + sqrtlinear_weight * sqrt(sqrtlinear^T x)
  s.t. x belongs to the combinatorial set defined by instance

Several implementations are provided, and can be chosen with the appropriate subtype of 
`ESCB2OptimisationAlgorithm`.

Returns one (approximately) optimum solution. The exact guarantees depend on the 
chosen algorithm. If `with_trace=true`, a second value is returned with comprehensive
details of the behaviour of the algorithm.
"""
function optimise_linear_sqrtlinear(::CombinatorialInstance{T}, 
                                    ::ESCB2OptimisationAlgorithm,
                                    ::Dict{T, Float64}, ::Dict{T, Float64}, ::Int,
                                    ::Float64) where T end

mutable struct ESCB2 <: Policy
  algo::ESCB2OptimisationAlgorithm
  confidence_bonus_scaling::Function
end

# Theoretical formula: log(n) + 4 m log(log(n)).
ESCB2(algo::ESCB2OptimisationAlgorithm) = ESCB2(algo, (state::State) -> log(state.round) / 2)
ESCB2(algo::ESCB2OptimisationAlgorithm, α::Float64) = ESCB2(algo, (state::State) -> α * log(state.round))

mutable struct ESCB2Details <: PolicyDetails
  n_iterations::Int
  best_objective::Float64
  solver_time::Float64

  function ESCB2Details()
    return new(0, 0.0, 0.0)
  end
end

function _escb2_confidence_values(policy::ESCB2, state::State{T}) where T
  weights = Dict(k => 1 / v for (k, v) in state.arm_counts)
  return weights, max(0, policy.confidence_bonus_scaling(state))
end

# TODO: shouldn't it go to a more generic function? Many algorithm work with indices, so maybe there is something to do... but only for black-box algorithms?
function escb2_index(policy::ESCB2, state::State{T}, solution::Vector{T}) where T
  w = state.arm_average_reward
  s2, f_n = _escb2_confidence_values(policy, state)
  return escb2_index(w, s2, f_n, solution)
end
function escb2_index(linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64}, sqrtlinear_weight::Float64, solution::Vector{T}) where T
  if length(solution) == 0
    return 0.0
  end

  avg_reward = sum(linear[arm] for arm in keys(linear) if arm in solution)
  confidence = sum(sqrtlinear[arm] for arm in keys(sqrtlinear) if arm in solution)
  return avg_reward + sqrt(sqrtlinear_weight * confidence)
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
  s2, f_n = _escb2_confidence_values(policy, state)

  # Solve the maximisation w^T x + \sqrt{f(n) s2^T x} with the required algorithm.
  return optimise_linear_sqrtlinear(instance, policy.algo, w, s2, f_n, state.round, with_trace=with_trace)
end
