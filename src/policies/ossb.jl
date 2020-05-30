# OSSB (optimal sampling for structured bandits).
# Based on https://arxiv.org/abs/1711.00400.

abstract type OSSBOptimisationAlgorithm end
function optimise_ossb(instance::CombinatorialInstance{T}, ::OSSBOptimisationAlgorithm,
                       weights::Dict{T, Float64}) where T end

mutable struct OSSB <: Policy
  algo::OSSBOptimisationAlgorithm
end

mutable struct OSSBDetails <: PolicyDetails
  n_iterations::Int
  solver_time::Float64

  function OSSBDetails()
    return new(0, 0.0)
  end
end

function choose_action(instance::CombinatorialInstance{T}, policy::ESCB2, state::State{T}; with_trace::Bool=false) where T
  # For initialisation, if some arms have never been tried, force them to be tried.
  # This should not be required for OSSB, but it improves running times for the first few iterations.
  if any(v == 0 for v in values(state.arm_counts))
    weights = Dict(arm => iszero(state.arm_counts[arm]) ? 1.0 : 0.0 for arm in keys(state.arm_counts))

    t0 = time_ns()
    sol = solve_linear(instance, weights)
    t1 = time_ns()

    if with_trace
      run_details = OSSBDetails()
      run_details.n_iterations = 1
      run_details.solver_time = (t1 - t0) / 1_000_000_000
      return sol, run_details
    else
      return sol
    end
  end

  # Solve OSSB's problem.
  return optimise_ossb(instance, policy.algo, state.arm_average_reward, with_trace=with_trace)
end
