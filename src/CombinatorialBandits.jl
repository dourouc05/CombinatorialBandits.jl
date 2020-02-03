module CombinatorialBandits
  using LinearAlgebra

  using DataStructures
  using IterTools
  using Distributions

  import Munkres # Avoid clashing with Hungarian.
  import Hungarian
  using LightGraphs
  using JuMP

  import Base: push!, copy, hash, isequal
  import JuMP: value

  export Policy, CombinatorialInstance, State,
         initial_state, initial_trace, simulate, choose_action, pull, update!, solve_linear, solve_budgeted_linear, solve_all_budgeted_linear, get_lp_formulation, has_lp_formulation, is_feasible, is_partially_acceptable,
         ThompsonSampling, LLR, CUCB, ESCB2, OLSUCB,
         ThompsonSamplingDetails, LLRDetails, CUCBDetails, ESCB2Details, OLSUCBDetails,
         ESCB2OptimisationAlgorithm, ESCB2Exact, ESCB2Greedy, ESCB2Budgeted, OLSUCBOptimisationAlgorithm, OLSUCBGreedy,
         PerfectBipartiteMatching, UncorrelatedPerfectBipartiteMatching, CorrelatedPerfectBipartiteMatching, PerfectBipartiteMatchingSolver, PerfectBipartiteMatchingLPSolver, PerfectBipartiteMatchingMunkresSolver, PerfectBipartiteMatchingHungarianSolver,
         ElementaryPath, ElementaryPathSolver, ElementaryPathLightGraphsDijkstraSolver, ElementaryPathLPSolver, ElementaryPathAlgosSolver,
         SpanningTree, SpanningTreeSolver, SpanningTreeLightGraphsPrimSolver, SpanningTreeAlgosSolver, SpanningTreeLPSolver,
         MSet, MSetSolver, MSetAlgosSolver, MSetLPSolver,
         # Algos.
         MSetInstance, MSetSolution, dimension, m, value, values, msets_greedy, msets_dp, msets_lp,
         BudgetedMSetInstance, BudgetedMSetSolution, weight, weights, budget, max_weight, items, items_all_budgets, budgeted_msets_dp, budgeted_msets_lp, budgeted_msets_lp_select, budgeted_msets_lp_all,
         ElementaryPathInstance, ElementaryPathSolution, graph, costs, src, dst, cost, lp_dp,
         BudgetedLongestPathInstance, BudgetedLongestPathSolution, rewards, reward, budgeted_lp_dp,
         SpanningTreeInstance, SpanningTreeSolution, st_prim,
         BudgetedSpanningTreeInstance, BudgetedSpanningTreeSolution, BudgetedSpanningTreeLagrangianSolution, SimpleBudgetedSpanningTreeSolution, _budgeted_spanning_tree_compute_value, _budgeted_spanning_tree_compute_weight, st_prim_budgeted_lagrangian, st_prim_budgeted_lagrangian_search, _solution_symmetric_difference, _solution_symmetric_difference_size, st_prim_budgeted_lagrangian_refinement, st_prim_budgeted_lagrangian_approx_half,
         BipartiteMatchingInstance, BipartiteMatchingSolution, matching_hungarian, BudgetedBipartiteMatchingInstance, BudgetedBipartiteMatchingSolution, BudgetedBipartiteMatchingLagrangianSolution, SimpleBudgetedBipartiteMatchingSolution, matching_hungarian_budgeted_lagrangian, matching_hungarian_budgeted_lagrangian_search, matching_hungarian_budgeted_lagrangian_refinement, matching_hungarian_budgeted_lagrangian_approx_half

  # General algorithm.
  abstract type Policy end
  abstract type PolicyDetails end
  abstract type CombinatorialInstance{T} end

  # Define the state of a bandit (evolves at each round).
  mutable struct State{T}
    round::Int
    regret::Float64
    reward::Float64
    arm_counts::Dict{T, Int}
    arm_reward::Dict{T, Float64}
    arm_average_reward::Dict{T, Float64}
  end

  copy(s::State{T}) where T = State{T}(s.round, s.regret, s.reward, s.arm_counts, s.arm_reward, s.arm_average_reward)

  # Define the trace of the execution throughout the rounds.
  struct Trace{T}
    states::Vector{State{T}}
    arms::Vector{Vector{T}}
    reward::Vector{Vector{Float64}}
    policy_details::Vector{PolicyDetails}
    time_choose_action::Vector{Int}
  end

  """
      push!(trace::Trace{T}, state::State{T}, arms::Vector{T}, reward::Vector{Float64}, policy_details::PolicyDetails, time_choose_action::Int) where T

  Appends the arguments to the execution trace of the bandit algorithm. More specifically, `trace`'s data structures are
  updated to also include `state`, `arms`, `reward`, `policy_details`, and `time_choose_action` (expressed in milliseconds).
  All of these arguments are copied, *except* `policy_details`.
  (Indeed, the usual scenario is to keep updating the state, the arms and the rewards, but to build the details at each round from the ground up.)
  """
  function push!(trace::Trace{T}, state::State{T}, arms::Vector{T}, reward::Vector{Float64}, policy_details::PolicyDetails, time_choose_action::Int) where T
    push!(trace.states, copy(state))
    push!(trace.arms, copy(arms))
    push!(trace.reward, copy(reward))
    push!(trace.policy_details, policy_details)
    push!(trace.time_choose_action, time_choose_action)
  end

  # Interface for combinatorial instances.
  function initial_state(instance::CombinatorialInstance{T}) where T end # TODO: Can't this be provided by default based on template types?
  function initial_trace(instance::CombinatorialInstance{T}) where T end # TODO: Can't this be provided by default based on template types?
  function is_feasible(instance::CombinatorialInstance{T}, arms::Vector{T}) where T end
  function is_partially_acceptable(instance::CombinatorialInstance{T}, arms::Vector{T}) where T end

  function all_arm_indices(reward::Matrix{Distribution})
    reward_indices_cartesian = eachindex(view(reward, [1:s for s in size(reward)]...))
    return [Tuple(i) for i in reward_indices_cartesian]
  end

  function all_arm_indices(reward::Vector{Distribution})
    return eachindex(view(reward, [1:s for s in size(reward)]...))
  end

  function all_arm_indices(reward::Dict{T, Distribution}) where T
    return collect(keys(reward))
  end

  function all_arm_indices(instance::CombinatorialInstance{T}) where T
    if isa(instance.reward, MultivariateDistribution) # Correlated arms.
      # Nothing as generic as the uncorrelated case is available, due to
      # the fact that only a vector is known, for any kind of arms (unlike
      # the uncorrelated case, where there is a distinction between vectors
      # and matrices of reward distributions).
      return instance.all_arm_indices
    else # Uncorrelated arms.
      return all_arm_indices(instance.reward)
    end
  end

  function pull(instance::CombinatorialInstance{T}, arms::Vector{T}) where T
    # Draw the rewards for this round. If T is a tuple, the reward distributions
    # are stored in a matrix, hence the splatting.
    arm_indices = all_arm_indices(instance)
    if isa(instance.reward, MultivariateDistribution) # Correlated arms.
      true_rewards_vector = mean(instance.reward)
      true_rewards = Dict{T, Float64}(arm => true_rewards_vector[i] for (i, arm) in enumerate(arm_indices))

      drawn_rewards_vector = rand(instance.reward)
      drawn_rewards = Dict{T, Float64}(arm => true_rewards_vector[i] for (i, arm) in enumerate(arm_indices))
    else # Uncorrelated arms.
      true_rewards = Dict{T, Float64}(i => mean(instance.reward[i...]) for i in arm_indices)
      drawn_rewards = Dict{T, Float64}(i => rand(instance.reward[i...]) for i in arm_indices)
    end

    # Select the information that will be provided back to the bandit policy.
    # Here is implemented the semi-bandit setting.
    true_reward = sum(true_rewards[arm] for arm in arms)
    reward = Float64[drawn_rewards[arm] for arm in arms]

    # Compute the incurred regret from the provided solution.
    incurred_regret = instance.optimal_average_reward - sum(true_reward)

    return reward, incurred_regret
  end

  solve_linear(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}) where T = solve_linear(instance.solver, rewards)
  solve_budgeted_linear(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, weights::Dict{T, Int}, budget::Int) where T =
    solve_budgeted_linear(instance.solver, rewards, weights, budget)
  solve_all_budgeted_linear(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, weights::Dict{T, Int}, max_budget::Int) where T =
    solve_all_budgeted_linear(instance.solver, rewards, weights, max_budget)
  has_lp_formulation(instance::CombinatorialInstance{T}) where T = has_lp_formulation(instance.solver)
  get_lp_formulation(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}) where T = has_lp_formulation(instance) ?
    get_lp_formulation(instance.solver, rewards) :
    error("The chosen solver uses no LP formulation.")

  # Implement the most common case. For tuples, the number vary more widely.
  function initial_state(instance::CombinatorialInstance{Int})
    n = instance.n_arms
    zero_counts = Dict(i => 0 for i in 1:n)
    zero_rewards = Dict(i => 0.0 for i in 1:n)
    return State{Int}(0, 0.0, 0.0, zero_counts, zero_rewards, copy(zero_rewards))
  end

  function initial_trace(instance::CombinatorialInstance{Int})
    return Trace{Int}(State{Int}[], Vector{Int}[], Vector{Float64}[], PolicyDetails[], Int[])
  end

  function initial_trace(instance::CombinatorialInstance{Tuple{Int, Int}})
    return Trace{Tuple{Int, Int}}(State{Tuple{Int, Int}}[], Vector{Tuple{Int, Int}}[], Vector{Float64}[], PolicyDetails[], Int[])
  end

  # Interface for policies.
  function choose_action(instance::CombinatorialInstance{T}, policy::Policy, state::State{T}) where T end

  # Update the state before the new round.
  function update!(state::State{T}, instance::CombinatorialInstance{T}, arms::Vector{T}, reward::Vector{Float64}, incurred_regret::Float64) where T
    state.round += 1

    # One reward per arm, i.e. semi-bandit feedback (not bandit feedback, where there would be only one reward for all arms).
    for i in 1:length(arms)
      state.arm_counts[arms[i]] += 1
      state.arm_reward[arms[i]] += reward[i]
      state.arm_average_reward[arms[i]] = state.arm_reward[arms[i]] / state.arm_counts[arms[i]]
      state.reward += reward[i]
    end

    state.regret += incurred_regret
  end

  # Use the bandit for the given number of steps.
  function simulate(instance::CombinatorialInstance{T}, policy::Policy, steps::Int; with_trace::Bool=false) where T
    state = initial_state(instance)
    if with_trace
      trace = initial_trace(instance)
    end

    for i in 1:steps
      t0 = time_ns()
      if with_trace
        arms, run_details = choose_action(instance, policy, state, with_trace=true)
      else
        arms = choose_action(instance, policy, state, with_trace=false)
      end
      t1 = time_ns()

      if length(arms) == 0
        error("No arms have been chosen at round $(i)!")
      end

      reward, incurred_regret = pull(instance, arms)
      update!(state, instance, arms, reward, incurred_regret)

      if with_trace
        push!(trace, state, arms, reward, run_details, round(Int, (t1 - t0) / 1_000_000_000))
      end

      if i % 100 == 0
        println(i)
      end
    end

    if ! with_trace
      return state
    else
      return state, trace
    end
  end

  ## Combinatorial algorithms. TODO: put this in another package.
  include("algos/helpers.jl")
  include("algos/lp.jl")
  include("algos/lp_budgeted.jl")
  include("algos/matching.jl")
  include("algos/matching_budgeted.jl")
  include("algos/msets.jl")
  include("algos/msets_budgeted.jl")
  include("algos/st.jl")
  include("algos/st_budgeted.jl")

  ## Bandit policies.
  include("policies/thompson.jl")
  include("policies/llr.jl")
  include("policies/cucb.jl")
  include("policies/escb2.jl")
  include("policies/olsucb.jl")

  include("policies/escb2_exact.jl")
  include("policies/escb2_greedy.jl")
  include("policies/escb2_budgeted.jl")

  include("policies/olsucb_greedy.jl")

  ## Potential problems to solve.
  include("instances/perfectbipartitematching.jl")
  include("instances/perfectbipartitematching_lp.jl")
  include("instances/perfectbipartitematching_munkres.jl")
  include("instances/perfectbipartitematching_hungarian.jl")
  # Not using LightGraphsMatching, due to BlossomV dependency (hard to get to work…)

  include("instances/elementarypath.jl")
  include("instances/elementarypath_algos.jl")
  include("instances/elementarypath_lightgraphsdijkstra.jl")
  include("instances/elementarypath_lp.jl")

  include("instances/spanningtree.jl")
  include("instances/spanningtree_algos.jl")
  include("instances/spanningtree_lightgraphsprim.jl")
  include("instances/spanningtree_lp.jl")

  include("instances/mset.jl")
  include("instances/mset_algos.jl")
  include("instances/mset_lp.jl")
end
