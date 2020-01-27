struct OLSUCBGreedy <: OLSUCBOptimisationAlgorithm end

function optimise_linear_sqrtquadratic(instance::CombinatorialInstance{T}, ::OLSUCBGreedy,
                                       linear::Dict{T, Float64}, sqrtquadratic::Dict{Tuple{T, T}, Float64},
                                       ε::Float64, d::Int, verbose::Int; with_trace::Bool=false) where T
  directions = Set(keys(linear))
  sol = T[]

  # As long as possible, add new arms to the solution to improve the expected reward.
  t0 = time_ns()
  n_iter = 0
  while true
    n_iter += 1

    # Pick the best direction to improve the reward.
    best_reward = 0.0
    best_direction = nothing
    for d in directions
      new_sol = copy(sol)
      push!(new_sol, d)
      if ! is_partially_acceptable(instance, new_sol)
        continue
      end

      reward = sum(linear[arm] for arm in keys(linear) if arm in new_sol) +
               sqrt(sum(sqrtquadratic[arms] for arms in keys(sqrtquadratic) if arms[1] in new_sol && arms[2] in new_sol))

      if reward > best_reward
        best_reward = reward
        best_direction = d
      end
    end

    # If no direction could be found, done!
    if best_direction == nothing
      break
    end

    # Update the solution and ensure this direction will never be tried again.
    push!(sol, best_direction)
    delete!(directions, best_direction)

    # If there is no more a direction to explore, done!
    if isempty(directions)
      break
    end
  end
  t1 = time_ns()

  if with_trace
    run_details = ESCB2Details()
    run_details.nIterations = n_iter
    run_details.bestLinearObjective = sum(linear[arm] + sqrtlinear[arm] for arm in keys(linear) if in(arm, sol))
    run_details.bestNonlinearObjective = sum(linear[arm] for arm in keys(linear) if in(arm, new_sol)) +
                                        sqrt(sum(sqrtquadratic[arms] for arms in keys(sqrtquadratic) if in(arms[1], new_sol) && in(arms[2], new_sol)))
    run_details.solverTimes = Float64[(t1 - t0) / 1_000_000_000]
    return sol, run_details
  else
    return sol
  end
end
