abstract type MSetSolver end
function build!(::MSetSolver, ::Matrix{Distribution})
  nothing
end

struct MSet <: CombinatorialInstance{Int}
  # Mandatory properties.
  n_arms::Int
  optimal_average_reward::Float64
  m::Int

  # Probability distributions for the arm rewards.
  reward::Vector{Distribution}

  # Internal solver.
  solver::MSetSolver

  function MSet(reward::Vector{Distribution}, m::Int, solver::MSetSolver)
    # Basically, any combination of parameters is allowable, even if m is
    # larger than the number of arms.

    # Prepare the solver to be used (if required).
    build!(solver, m, length(reward))

    avg_reward = Dict{Int, Float64}(i => mean(reward[i...]) for i in all_arm_indices(reward))
    opt_sol = solve_linear(solver, avg_reward)
    opt = sum(avg_reward[i] for i in opt_sol)

    # Done!
    return new(length(reward), opt, m, reward, solver)
  end
end

copy(instance::MSet) =
  MSet(instance.reward, instance.m, copy(instance.solver))

function is_feasible(instance::MSet, arms::Vector{Int})
  if length(arms) > instance.m
    return false
  end

  if length(arms) != length(unique(arms))
    return false
  end

  return true
end

function is_partially_acceptable(instance::MSet, arms::Vector{Int})
  return is_feasible(instance, arms)
end
