# OSSB (optimal sampling for structured bandits).
# Based on https://arxiv.org/abs/1711.00400.

abstract type OSSBOptimisationAlgorithm end

function optimise_graves_lai(::CombinatorialInstance{T}, algo::OSSBOptimisationAlgorithm, ::Dict{T, Float64}; kwargs...) where T
  error("OSSB optimiser $algo has not yet been implemented.")
end

struct OSSB <: Policy
  algo::OSSBOptimisationAlgorithm
  γ::Float64
  ε::Float64
end

mutable struct GravesLaiResults{T}
  objective::Float64
  points::Vector{Vector{T}} # Vector of solutions.
  weights::Vector{Float64} # Vector of weights, one per solution.
  delayed::Union{Function, Nothing}
end

function GravesLaiResults(objective::Float64, points::Vector{Vector{T}}, weights::Vector{Float64}) where T
  return GravesLaiResults{T}(objective, points, weights, nothing)
end

function GravesLaiResults(instance::CombinatorialInstance{T}, t_val::Dict{T, Float64}, objective::Float64) where T
  return GravesLaiResults{T}(objective, Vector{Vector{T}}(undef, 0), Vector{Float64}(undef, 0),
                             () -> _ossb_exact_convex_combination(copy(instance), t_val))
end

function Base.getproperty(obj::GravesLaiResults, sym::Symbol)
  # If one of the delayed field is requested, start the computations.
  if getfield(obj, :delayed) !== nothing && (sym === :points || sym === :weights)
    obj.points, obj.weights = obj.delayed()
    obj.delayed = nothing
  end

  # Return the requested value.
  return getfield(obj, sym)
end


mutable struct GravesLaiDetails <: PolicyDetails
  n_iterations::Int
  time_per_iteration::Vector{Float64} # ms
  objectives::Vector{Float64}
  gl_bounds::Vector{Float64} # Usually equal to the objective, but not always: the objective may include a penalisation.
end

GravesLaiDetails(time::Float64, objective::Float64) = GravesLaiDetails(1, [time], [objective], [objective])
GravesLaiDetails() = GravesLaiDetails(0, Float64[0.0], Float64[0.0], Float64[0.0])

mutable struct OSSBDetails <: PolicyDetails
  solver_time::Float64 # ms
  graves_lai::GravesLaiDetails
end

OSSBDetails() = OSSBDetails(0.0, GravesLaiDetails())

function choose_action(instance::CombinatorialInstance{T}, policy::OSSB, state::State{T}; with_trace::Bool=false) where T
  # Ensure that the state contains a counter for the number of times each solution is played (and not every arm).
  # This requirement is very OSSB-specific.
  # Do not pre-populate this data structure: in the worst case, it contains one entry per combinatorial solution.
  # OSSB also requires a counter of rounds where there has been no exploitation.
  if ! (:ossb in state.policy_extension)
    state.policy_extension[:ossb] = (0, Dict{Vector{T}, Int}())
  end

  # For initialisation, if some arms have never been tried, force them to be tried.
  # This should not be required for OSSB, but it improves running times for the first few iterations.
  if any(v == 0 for v in values(state.arm_counts))
    weights = Dict(arm => iszero(state.arm_counts[arm]) ? 1.0 : 0.0 for arm in keys(state.arm_counts))

    t0 = time_ns()
    sol = solve_linear(instance, weights)
    t1 = time_ns()

    if with_trace
      return sol, OSSBDetails((t1 - t0) / 1_000_000, GravesLaiDetails())
    else
      return sol
    end
  end

  # Solve OSSB's problem to get a distribution over actions to play (not really a probability distribution, however).
  t0 = time_ns()
  if with_trace
    gl, gl_trace = optimise_graves_lai(instance, policy.algo, state.arm_average_reward, with_trace=true)
  else
    gl = optimise_graves_lai(instance, policy.algo, state.arm_average_reward, with_trace=false)
  end
  t1 = time_ns()

  # Sometimes, the (0, 0...) vertex is found. For bandits, this is very bad: no arm is played, no reward can be
  # obtained, no exploration is performed.
  if any(iszero.(gl.points))
    idx = findall(iszero.gl.(points))
    deleteat!(gl.points, idx)
    deleteat!(gl.weights, idx)
  end

  # If the solution is zero, there is not much to do. This probably indicates a bug in the underlying solver.
  if length(gl.points) == 0
    println("Uh ho...")
    println(gl.points)
    println(gl.weights)

    t0 = time_ns()
    sol = solve_linear(instance, state.arm_average_reward)
    t1 = time_ns()

    if with_trace
      return sol, OSSBDetails((t1 - t0) / 1_000_000, gl_trace)
    else
      return sol
    end
  end

  # Exploit phase? Check if all solutions have already been played enough according to the new solution.
  enters_exploit = true
  unplayed_solution = nothing
  for (idx, point) in enumerate(gl.points)
    # Has this solution ever been played?
    if !(point in keys(state.policy_extension[:ossb][1]))
      enters_exploit = false
      unplayed_solution = point
      break
    end

    # Has this solution been played enough?
    if gl.weights[idx] * (1 + policy.γ) * log(state.round) <= state.policy_extension[:ossb][1][point]
      enters_exploit = false
      break
    end

    # Ignore the other solutions, as they have a zero weight: they already have been played enough, as the current
    # solution recommends to play them at least zero times.
  end

  if enters_exploit
    t0 = time_ns()
    sol = solve_linear(instance, state.arm_average_reward)
    t1 = time_ns()

    if with_trace
      return sol, OSSBDetails((t1 - t0) / 1_000_000, gl_trace)
    else
      return sol
    end
  end

  # Explore or sample phase?
  state.policy_extension[:ossb][2] += 1 # One more round without exploitation.

  if unplayed_solution !== nothing
    # At least one solution has never been played: 0 <= ε S, so play it right now.
    # Special case of the explore phase.
    if with_trace
      return unplayed_solution, OSSBDetails(0.0, gl_trace)
    else
      return unplayed_solution
    end
  end

  least_played = reduce((x, y) -> state.policy_extension[:ossb][1][x] ≤ state.policy_extension[:ossb][1][y] ? x : y, keys(state.policy_extension[:ossb][1]))

  # Explore.
  # This search can be limited to the points in the returned convex combination: if another solution was selected as
  # least played, the comparison would reduce to 0 <= something.
  if gl.weights[least_played] * (1 + policy.γ) * log(state.round) <= state.policy_extension[:ossb][1][least_played]
    if with_trace
      return least_played, OSSBDetails(0.0, gl_trace)
    else
      return least_played
    end
  end

  # Sample.
  # Again, this search can be limited to the points in the returned convex combination: the other solutions would
  # be associated to either 0/0 or x/0 with x > 0, and these values are either undefined or infinite.
  furthest_away = nothing
  furthest_away_obj = Inf

  t0 = time_ns()
  for (idx, point) in enumerate(gl.points)
    obj = state.policy_extension[:ossb][1][point] / gl.weights[idx]
    if obj <= furthest_away_obj
      furthest_away = point
      furthest_away_obj = obj
    end
  end
  t1 = time_ns()

  if with_trace
    return least_played, OSSBDetails((t1 - t0) / 1_000_000, gl_trace)
  else
    return least_played
  end
end
