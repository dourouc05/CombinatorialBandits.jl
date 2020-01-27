# OLS-UCB, "ordinary least squares, upper confidence bound".
# Based on https://arxiv.org/abs/1612.01859.

abstract type OLSUCBOptimisationAlgorithm end
function optimise_linear_sqrtquadratic(instance::CombinatorialInstance{T}, ::OLSUCBOptimisationAlgorithm,
                                       linear::Dict{T, Float64}, sqrtquadratic::Dict{Tuple{T, T}, Float64},
                                       ε::Float64, d::Int, verbose::Int; with_trace::Bool=false) where T end

mutable struct OLSUCB <: Policy
  ε::Float64 # Only for approximation algorithm. TODO: Move, like ESCB-2?
  λ::Float64 # Numerical parameter.
  Γ::Matrix{Float64} # Approximation of the covariance matrix.
  algo::OLSUCBOptimisationAlgorithm
  verbose::Int # 0: nothing. 1: summary of function. 2: iteration counter too.

  function OLSUCB(ε::Float64, λ::Float64, Γ::Matrix{Float64}, algo::OLSUCBOptimisationAlgorithm, verbose::Int=0)
    if any(Γ .< 0)
      error("The approximate covariance matrix must have nonnegative entries")
    end

    if ! isposdef(Γ)
      error("The approximate covariance matrix must be positive definite")
    end

    # Elements outside the diagonal are bounded.
    for i in 1:size(Γ, 1)
      for j in 1:size(Γ, 2)
        if i != j
          if Γ[i, j] > sqrt(Γ[i, i] * Γ[j, j])
            error("The approximate covariance matrix does not respect the bounding property Γij ≤ √(Γii Γjj) for i = $i and j = $j")
          end
        end
      end
    end

    if λ <= 0.
      error("The lambda (λ) parameter must be strictly positive")
    end

    return new(ε, λ, Γ, algo, verbose)
  end
end

mutable struct OLSUCBDetails <: PolicyDetails
  nIterations::Int
  nBreakpoints::Int
  bestLambda::Float64
  bestLinearObjective::Float64
  bestNonlinearObjective::Float64
  solverTimes::Vector{Float64}

  function OLSUCBDetails()
    return new(0, 0, 0.0, 0.0, 0.0, Float64[])
  end
end

function choose_action(instance::CombinatorialInstance{T}, policy::OLSUCB, state::State{T}; with_trace::Bool=false) where T
  # For initialisation, if some arms have never been tried, force them to be tried.
  # Without this, the standard deviation term cannot exist.
  if any(v == 0 for v in values(state.arm_counts))
    weights = Dict(arm => (state.arm_counts[arm] == 0.0) ? 1.0 : 0.0 for arm in keys(state.arm_counts))

    t0 = time_ns()
    sol = solve_linear(instance, weights)
    t1 = time_ns()

    if with_trace
      run_details = OLSUCBDetails()
      push!(run_details.solverTimes, (t1 - t0) / 1_000_000_000)
      return sol, run_details
    else
      return sol
    end
  end

  # If all arms have been tried, can compute the two parts of the objective function: the average reward and a standard deviation.
  t_dict = state.arm_counts
  w_dict = state.arm_average_reward
  n = state.round

  arms = collect(keys(t_dict))
  t = Float64[t_dict[arm] for arm in arms]
  w = Float64[w_dict[arm] for arm in arms]

  d = length(w)
  fn = log(n) + (d + 2) * log(log(n)) + d / 2 * log(1 + e / policy.λ)
  S = 2 * fn * inv(Diagonal(t)) * (policy.λ * Diagonal(policy.Γ) * Diagonal(t) + Diagonal(t) * Diagonal(policy.Γ)) * inv(Diagonal(t))

  S_dict = Dict{Tuple{T, T}, Float64}((arms[i], arms[j]) => S[i, j] for (i, j) in product(1:length(arms), 1:length(arms)))

  # Solve the maximisation w^T x + \sqrt{x^T S x} based on the chosen algorithm.
  return optimise_linear_sqrtquadratic(instance, policy.algo, w_dict, S_dict, policy.ε, d, policy.verbose, with_trace=with_trace)
end
