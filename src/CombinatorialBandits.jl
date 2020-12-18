# TODO: switch the inferface from Vector{arms} to Set{arms}? For now, order might still be important... but it should not!
# TODO: have a look at https://github.com/KristofferC/TimerOutputs.jl for timing.
# TODO: switch to the generic solve() interface of Kombinator, THEN USE THE PACKAGE'S approximation_ratio/term!!!!!!
# TODO: refactor the instances, an LP formulation is a property of the problem, not of an algorithm. (Still need an extra abstraction layer, because there may be several good formulations for a given problem.)

module CombinatorialBandits
  using LinearAlgebra
  using SparseArrays

  using DataStructures
  using IterTools
  using Distributions

  import Munkres # Avoid clashing with Hungarian: both packages propose a munkres() function.
  import Hungarian
  using LightGraphs
  using JuMP

  import Base: push!, copy, hash, isequal
  import JuMP: value

  ## Combinatorial algorithms. TODO: put this in another package.
  include("algos/Kombinator.jl")
  using .Kombinator
  import .Kombinator: dimension, value

  ## Nonsmooth optimisation. TODO: put this in another package.
  include("nso/NonsmoothOptim.jl")
  using .NonsmoothOptim

  # General algorithm.
  abstract type Policy end
  abstract type PolicyDetails end

  """
      CombinatorialInstance{T}

  A combinatorial instance, also called an environment in reinforcement learning. An instance is the combination of a
  combinatorial structure (i.e. what the bandit is allowed to make in the environment) and of the statistical
  properties of the arms (i.e. what reward the bandit gets).

  Two properties are mandatory:

  * `n_arms`: the number of arms that are available in the instance (i.e. the dimension of the problem)
  * `optimal_average_reward`: the best reward that can be obtained at any round, in expectation
  """
  abstract type CombinatorialInstance{T} end

  """
      State{T}

  Memorise the (global) evolution of the bandit in the environment. This structure mostly contains information required
  by most bandit algorithms to decide their next step.

  * `round`: the number of rounds (time steps) the bandit has already played
  * `regret`: the total regret the bandit faced (i.e. how much more reward it would have got if it always played the
    optimum solution)
  * `reward`: the total reward the bandit managed to gather
  * `arm_counts`: the number of times each arm (and not each solution, which is a set of arms)  has been played.
    This information is used by most bandit algorithms
  * `arm_reward`: the total reward obtained by this arm
  * `arm_average_reward`: the average reward obtained by this arm. This information is used by most bandit algorithms
  * `policy_extension`: if a policy needs more information than this structure contains, it can store, in a key that
    uniquely identifies the policy, anything it might require
  """
  mutable struct State{T}
    round::Int
    regret::Float64
    reward::Float64
    arm_counts::Dict{T, Int}
    arm_reward::Dict{T, Float64}
    arm_average_reward::Dict{T, Float64}
    policy_extension::Dict{Symbol, Any}
  end

  copy(s::State{T}) where T = State{T}(s.round, s.regret, s.reward, copy(s.arm_counts), copy(s.arm_reward),
                                       copy(s.arm_average_reward), copy(s.policy_extension))

  # Define the trace of the execution throughout the rounds. The structure itself is immutable, things only get appended.
  """
      Trace{T}

  Memorise all the details about the evolution of the bandit in the environment. This object can therefore become
  very heavy for long episodes.

  The information is stored in several vectors, all of them having the same length equal to the number of rounds the
  bandit was used:

  * `states`: a list of states the bandit entered into over the episode
  * `arms`: a list of sets of arms that have been played over the episode
  * `reward`: a list of reward the policy received over the episode
  * `policy_details`: a list of policy-defined details over the episode (typically, computation times)
  * `time_choose_action`: a list of times (in milliseconds) to take a round decision over the episode
  """
  struct Trace{T}
    states::Vector{State{T}}
    arms::Vector{Vector{T}}
    reward::Vector{Vector{Float64}}
    policy_details::Vector{PolicyDetails}
    time_choose_action::Vector{Int}
  end

  """
      push!(trace::Trace{T}, state::State{T}, arms::Vector{T}, reward::Vector{Float64}, policy_details::PolicyDetails,
            time_choose_action::Int) where T

  Appends the arguments to the execution trace of the bandit algorithm. More specifically, `trace`'s data structures
  are updated to also include `state`, `arms`, `reward`, `policy_details`, and `time_choose_action` (expressed in
  milliseconds).

  All of these arguments are copied, *except* `policy_details`. (Indeed, the usual scenario is to keep updating the
  state, the chosen solution (set of arms), and the reward objects, but to build the detail object at each round from
  the ground up.)
  """
  function push!(trace::Trace{T}, state::State{T}, arms::Vector{T}, reward::Vector{Float64},
                 policy_details::PolicyDetails, time_choose_action::Int) where T
    push!(trace.states, copy(state))
    push!(trace.arms, copy(arms))
    push!(trace.reward, copy(reward))
    push!(trace.policy_details, policy_details)
    push!(trace.time_choose_action, time_choose_action)
  end

  """
      all_arm_indices(reward)

  Returns a list of arm indices for the given reward. For instance, for a vector of arms, it returns the list of indices
  in that vector: each index is associated to an arm.

      all_arm_indices(instance::CombinatorialInstance{T}) where T

  Returns a list of arm indices for the given combinatorial instance.
  """
  function all_arm_indices end

  all_arm_indices(reward::Vector{Distribution}) = eachindex(view(reward, [1:s for s in size(reward)]...))
  all_arm_indices(reward::Dict{T, Distribution}) where T = collect(keys(reward))

  function all_arm_indices(reward::Matrix{Distribution})
    reward_indices_cartesian = eachindex(view(reward, [1:s for s in size(reward)]...))
    return [Tuple(i) for i in reward_indices_cartesian]
  end

  """
      initial_state(instance::CombinatorialInstance{Int})

  Returns a new empty `State` object for the given combinatorial instance.
  """
  function initial_state(instance::CombinatorialInstance{T}) where T
    zero_counts = Dict(i => 0 for i in all_arm_indices(instance))
    zero_rewards = Dict(i => 0.0 for i in all_arm_indices(instance))
    return State{T}(0, 0.0, 0.0, zero_counts, zero_rewards, copy(zero_rewards), Dict{Symbol, Any}())
  end

  """
      initial_state(instance::CombinatorialInstance{Int})

  Returns a new empty `Trace` object for the given combinatorial instance.
  """
  function initial_trace(::CombinatorialInstance{T}) where T
    return Trace{T}(State{T}[], Vector{T}[], Vector{Float64}[], PolicyDetails[], Int[])
  end

  """
      dimension(instance::CombinatorialInstance{T}) where T

  Returns the dimension of a solution.
  """
  function dimension(instance::CombinatorialInstance{T}) where T
    return instance.n_arms
  end

  """
      is_feasible(instance::CombinatorialInstance{T}, arms::Vector{T})

  Returns whether the set of arms is a solution that can be played for the given combinatorial instance.
  """
  function is_feasible(::CombinatorialInstance{T}, ::Vector{T}) where T end

  """
      is_partially_acceptable(instance::CombinatorialInstance{T}, arms::Vector{T})

  Returns whether the set of arms is either a solution or the subset of a solution that can be played for the given
  combinatorial instance. In some cases, a subset of a solution is not a solution: the distinction between the two
  states is made by calling `is_feasible`.

  Typically, `is_partially_acceptable` returns `true` for an empty set of arms: even if this is not an acceptable
  solution, adding elements to the empty set may yield a perfectly acceptable solution.
  """
  function is_partially_acceptable(::CombinatorialInstance{T}, ::Vector{T}) where T end

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

  """
      pull(instance::CombinatorialInstance{T}, arms::Vector{T}) where T

  For the given bandit, plays the given solution (i.e. set of arms). This function returns both the reward and the
  regret this action caused (zero if the action is optimum, greater than zero otherwise).
  """
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
  supports_solve_budgeted_linear(instance::CombinatorialInstance{T}) where T = supports_solve_budgeted_linear(instance.solver)
  solve_all_budgeted_linear(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}, weights::Dict{T, Int}, max_budget::Int) where T =
    solve_all_budgeted_linear(instance.solver, rewards, weights, max_budget)
  supports_solve_all_budgeted_linear(instance::CombinatorialInstance{T}) where T = supports_solve_budgeted_linear(instance.solver)
  has_lp_formulation(instance::CombinatorialInstance{T}) where T = has_lp_formulation(instance.solver)
  get_lp_formulation(instance::CombinatorialInstance{T}, rewards::Dict{T, Float64}) where T = has_lp_formulation(instance) ?
    get_lp_formulation(instance.solver, rewards) :
    error("The chosen solver uses no LP formulation.")
  approximation_ratio(instance::CombinatorialInstance{T}) where T = approximation_ratio(instance.solver)
  approximation_term(instance::CombinatorialInstance{T}) where T = approximation_term(instance.solver)
  approximation_ratio_budgeted(instance::CombinatorialInstance{T}) where T = approximation_ratio_budgeted(instance.solver)
  approximation_term_budgeted(instance::CombinatorialInstance{T}) where T = approximation_term_budgeted(instance.solver)

  # Interface for policies.
  function choose_action(::CombinatorialInstance{T}, policy::Policy, ::State{T}) where T
    error("Policy not implemented: $policy")
  end

  # Update the state before the new round.
  function update!(state::State{T}, ::CombinatorialInstance{T}, arms::Vector{T}, reward::Vector{Float64}, incurred_regret::Float64) where T
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

  ## Common estimation procedures.
  function estimate_Δmax(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}) where T # TODO: expensive. Memoisation?
    x_max = solve_linear(copy(instance), θ)
    r_max = sum(θ[i] for i in x_max)
    x_min = solve_linear(copy(instance), Dict(k => -v for (k, v) in θ))
    r_min = (length(x_min) > 0) ? sum(θ[i] for i in x_min) : 0.0
    Δmax = r_max - r_min
    return Δmax
  end

  function estimate_Δmin(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}) where T # TODO: expensive. Memoisation?
    x_max = solve_linear(copy(instance), θ) # TODO: redundant with estimate_Δmax. Memoisation?
    r_max = sum(θ[arm] for arm in x_max)

    Δmin = Inf
    for i in keys(θ)
      new_weights = copy(θ)
      if i in x_max
        new_weights[i] = 0.0 # Remove the element.
      else
        new_weights[i] = dimension(instance) * maximum(values(θ)) # Force the element to be taken.
      end
      new_x = solve_linear(instance, new_weights)
      new_r = sum(θ[i] for i in new_x)

      # Focus on suboptimal solutions: Δmin > 0 even if there are multiple optimal solutions.
      if r_max == new_r
        continue
      end

      # Finally, update Δmin.
      if r_max - new_r < Δmin
        Δmin = r_max - new_r
      end
    end
    return Δmin
  end

  function maximum_solution_length(instance::CombinatorialInstance{T}) where T # Common symbol: m; too common, high risk of clash.
    # TODO: expensive but very very common. Memoisation?
    weights = Dict{T, Float64}(k => 1.0 for k in all_arm_indices(instance))
    return length(solve_linear(instance, weights))
  end

  ## Helpers.
  include("budgeted.jl")

  ## Bandit policies.
  include("policies/thompson.jl")
  include("policies/llr.jl")
  include("policies/cucb.jl")
  include("policies/escb2.jl")
  include("policies/olsucb.jl")
  include("policies/ossb.jl")

  include("policies/escb2_exact.jl")
  include("policies/escb2_greedy.jl")
  include("policies/escb2_budgeted.jl")

  include("policies/ossb_helpers.jl")
  include("policies/ossb_convexcombination.jl")
  include("policies/ossb_exact.jl")
  include("policies/ossb_exactsmart.jl")
  include("policies/ossb_exactnaive.jl")
  include("policies/ossb_subgradient_budgeted.jl")

  include("policies/olsucb_greedy.jl")

  ## Potential problems to solve.
  include("instances/_lp_generic.jl")

  include("instances/perfectbipartitematching.jl")
  include("instances/perfectbipartitematching_algos.jl")
  include("instances/perfectbipartitematching_lp.jl")
  include("instances/perfectbipartitematching_munkres.jl")
  include("instances/perfectbipartitematching_hungarian.jl")
  # Not using LightGraphsMatching, due to BlossomV dependency (hard to get to work on Windows, especially…)

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

  # Export all symbols. Code copied from JuMP.
  const _EXCLUDE_SYMBOLS = [Symbol(@__MODULE__), :eval, :include]

  for sym in names(@__MODULE__, all=true)
    sym_string = string(sym)
    if sym in _EXCLUDE_SYMBOLS || startswith(sym_string, "_")
        continue
    end
    if !(Base.isidentifier(sym) || (startswith(sym_string, "@") &&
         Base.isidentifier(sym_string[2:end])))
       continue
    end
    @eval export $sym
  end
end
