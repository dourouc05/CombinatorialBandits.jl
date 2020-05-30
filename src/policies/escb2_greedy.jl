struct ESCB2Greedy <: ESCB2OptimisationAlgorithm end

function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, ::ESCB2Greedy,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64},
                                    sqrtlinear_weight::Float64, ::Int;
                                    with_trace::Bool=false) where T
  # TODO: factor this out to the combinatorial algorithm package? This is very generic, in principle (get rid of linear and sqrtlinear parts, replace by a custom evaluation function).
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

      # If this solution is better, keep it.
      reward = escb2_index(linear, sqrtlinear, sqrtlinear_weight, new_sol)
      if reward > best_reward
        best_reward = reward
        best_direction = d
      end
    end

    # If no direction could be found, done!
    if best_direction === nothing
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

  # Sometimes, this basic algorithm cannot find a feasible solution, even though
  # it only traversed partially feasible solutions.
  if ! is_feasible(instance, sol)
    sol = T[]
  end

  if with_trace
    run_details = ESCB2Details()
    run_details.n_iterations = n_iter
    run_details.best_objective = escb2_index(linear, sqrtlinear, sqrtlinear_weight, sol)
    run_details.solver_time = (t1 - t0) / 1_000_000_000
    return sol, run_details
  else
    return sol
  end
end
