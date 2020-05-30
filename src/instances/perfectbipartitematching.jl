# Perfect matching in complete bipartite graphs.

## Components shared by both correlated and uncorrelated arms.

abstract type PerfectBipartiteMatchingSolver end
function build!(::PerfectBipartiteMatchingSolver, ::Int) end

abstract type PerfectBipartiteMatching <: CombinatorialInstance{Tuple{Int, Int}} end

function is_feasible(instance::PerfectBipartiteMatching, arms::Vector{Tuple{Int, Int}})
  if length(arms) > instance.n_arms
    return false
  end

  # Check whether each node is taken at most once on each side.
  lefts = sort([arm[1] for arm in arms])
  rights = sort([arm[2] for arm in arms])

  # No duplicates on either side.
  return lefts == unique(lefts) && rights == unique(rights)
end

# Any partial matching is feasible.
is_partially_acceptable(instance::PerfectBipartiteMatching, arms::Vector{Tuple{Int, Int}}) = is_feasible(instance, arms)

## Uncorrelated arms.

struct UncorrelatedPerfectBipartiteMatching <: PerfectBipartiteMatching
  # Mandatory properties.
  n_arms::Int
  optimal_average_reward::Float64

  # Probability distributions for the arm rewards.
  reward::Matrix{Distribution}

  # Internal solver.
  solver::PerfectBipartiteMatchingSolver

  function UncorrelatedPerfectBipartiteMatching(reward::Matrix{Distribution}, solver::PerfectBipartiteMatchingSolver)
    n = size(reward, 1)
    if n != size(reward, 2)
      error("Graph is not bipartite complete: reward matrix must be square.")
    end

    # Prepare the solver to be used (if required).
    build!(solver, n)

    avg_reward = Dict{Tuple{Int, Int}, Float64}(i => mean(reward[i...]) for i in all_arm_indices(reward))
    opt_sol = solve_linear(solver, avg_reward)
    opt = sum(avg_reward[i] for i in opt_sol)

    # Done!
    return new(n ^ 2, opt, reward, solver)
  end
end

Base.copy(instance::UncorrelatedPerfectBipartiteMatching) = UncorrelatedPerfectBipartiteMatching(instance.reward, copy(instance.solver))

## Correlated arms.

struct CorrelatedPerfectBipartiteMatching <: PerfectBipartiteMatching
  # Mandatory properties.
  n_arms::Int
  optimal_average_reward::Float64
  all_arm_indices::Vector{Tuple{Int, Int}}

  # Probability distributions for the arm rewards.
  reward::MultivariateDistribution

  # Internal solver.
  solver::PerfectBipartiteMatchingSolver

  """
  Builds a perfect bipartite matching instance where rewards can be correlated.

  * `reward` is the distribution of the rewards, which can be as complex as required.
    It is supposed to return a vector of size \$\\left(\\# arms\\right)^2\$, with
    element \$\\left(\\# arms \\times (i-1)\\right) + j\$ corresponds to choosing the
    matching between \$i\$ and \$j\$.
  * `solver` is a perfect bipartite matching solver. Nothing is specific to correlated
    matching at this point: the only difference is in the bandit algorithm.
  """
  function CorrelatedPerfectBipartiteMatching(reward::MultivariateDistribution, solver::PerfectBipartiteMatchingSolver)
    n = length(reward)
    n_arms = round(Int, sqrt(n))
    all_arm_indices = vec(Tuple{Int, Int}[(i, j) for i in 1:n_arms, j in 1:n_arms])

    # Prepare the solver to be used (if required).
    build!(solver, n_arms)

    avg_reward_vector = mean(reward)
    avg_reward = Dict{Tuple{Int, Int}, Float64}(i => v for (v, i) in zip(avg_reward_vector, all_arm_indices))
    opt_sol = solve_linear(solver, avg_reward)
    opt = sum(avg_reward[i] for i in opt_sol)

    # Done!
    return new(n, opt, all_arm_indices, reward, solver)
  end
end

Base.copy(instance::CorrelatedPerfectBipartiteMatching) = CorrelatedPerfectBipartiteMatching(instance.reward, copy(instance.solver))
