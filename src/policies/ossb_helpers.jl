function _ossb_build_basis_of_solutions(instance::CombinatorialInstance{T}) where T
  # "Basis": the i-th returned vector must use arm i. The same solution might be returned for 
  # several arms.
  d = dimension(instance)
  x = Vector{Vector{T}}(undef, d)
  all_arms = all_arm_indices(instance)
  for arm_to_include in 1:d
    # TODO: rather use DataStructures.jl's DefaultDict, but CombinatorialBandits.jl sometimes uses keys(weights) for a set of arms... Rather use all_arm_indices(instance) in a next refactoring.
    weights = Dict{T, Float64}()
    for (arm, arm_object) in enumerate(all_arms)
      weights[arm_object] = (arm == arm_to_include) ? 1.0 : 0.0
    end
    x[arm_to_include] = solve_linear(instance, weights)
  end
  return x
end

function _ossb_initial_feasible_solution(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}) where T
  # Compute useful results. 
  d = dimension(instance)
  m = maximum_solution_length(instance)
  Δmin = estimate_Δmin(instance, θ)
  idx_to_arm = Dict(idx => arm for (idx, arm) in enumerate(all_arm_indices(instance)))
  x = _ossb_build_basis_of_solutions(instance)

  # Ignore repeated solutions in the basis (loses an indexing property, but that's not important).
  x = collect(Set(x))

  # Compute the initial solution.
  v = d * m / Δmin^2
  t = zeros(d)
  for i in 1:d
    for j in 1:length(x)
      t[i] += any(x[j] .== Ref(idx_to_arm[i])) # Ref(.) to avoid broadcasting over the contents of solutions.
    end
  end

  return (t * v, v)
end

function _ossb_confusing_arms(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}; ε::Float64=1.0e-6) where T
  # Precompute useful results.
  d = dimension(instance)
  idx_to_arm = Dict(idx => arm for (idx, arm) in enumerate(all_arm_indices(instance)))
  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)

  # Compute the set of confusing arms.
  I = Set{T}()
  for i in 1:d
    # Compute a solution that plays arm i *and* maximises the reward.
    # In particular, this is not necessarily a solution computed by _ossb_build_basis_of_solutions, because that 
    # basis only ensures that an arm is included in the solution, not that it maximises the reward in any sense.
    weights = copy(θ)
    weights[idx_to_arm[i]] = d^2 * max(maximum(abs.(collect(values(θ)))), 1) # Force i to be part of the solution.
    xi = solve_linear(copy(instance), weights)

    if length(xi) == 0
      # Probably an error when optimising, skip.
      continue
    end

    # If this solution yields a suboptimum reward, then it cannot be part of an optimum solution.
    if sum(θ[arm] for arm in xi) <= r_max * (1.0 - ε)
      push!(I, idx_to_arm[i])
    end
  end

  return I
end

function _ossb_is_solution_feasible(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}, t::Vector{Float64}, solve_all_budgets_at_once::Bool) where T
  # Check whether the constraints ∑i x_i / t_i ≤ Δx² are respected. Procedure: maximise ∑i x_i / t_i - Δx², check the sign.
  Δmax = estimate_Δmax(instance, θ)
  Δmin = estimate_Δmin(instance, θ)
  arm_to_idx = Dict(arm => idx for (idx, arm) in enumerate(all_arm_indices(instance)))
  I = _ossb_confusing_arms(instance, θ)
  # I = all_arm_indices(instance)
  I_all = all_arm_indices(instance) # Only for gap.

  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)

  # Ensure no component of t is zero.
  t = copy(t)
  t[t .== 0.0] .= minimum(t[t .> 0.0])

  # Actual maximisation.
  # m, _, vars = get_lp_formulation(copy(instance), θ) # A t ≤ b
  # @objective(m, Min, (sum(vars[i] * θ[i] for i in I_all) - r_max) ^2 - sum(vars[i] / t[arm_to_idx[i]] for i in I))
  # # @objective(m, Max, sum(vars[i] / t[arm_to_idx[i]] for i in I) - (sum(vars[i] * θ[i] for i in I) - r_max) ^2)
  # @constraint(m, sum(vars[i] for i in I) >= 1)
  # optimize!(m)
  # best_objective = objective_value(m)
  # @show [value(vars[i]) for i in I]
  # @show best_objective
  # @show (sum(value(vars[i]) * θ[i] for i in I_all) - r_max) ^2

  # println(m)
  rewards = Dict{T, Float64}(i => 1 / t[arm_to_idx[i]] for i in I_all)
  weights = Dict{T, Int}(i => round(Int, θ[i] * t[arm_to_idx[i]] / Δmin, RoundUp) for i in I)
  # @show weights
  # @show Dict(i => θ[i] * t[arm_to_idx[i]] for i in I)
  _, o = _maximise_nonlinear_through_budgeted(instance, rewards, weights, round(Int, Δmax / Δmin), 
                                                           (sol, budget) -> begin 
                                                              # @show sol
                                                              # @show budget * Δmin
                                                              # @show (budget * Δmin - r_max)^2
                                                              sum(rewards[i] for i in sol) - (budget * Δmin - r_max)^2
                                                            end,
                                                           solve_all_budgets_at_once)

  # @show o

  # best_subobjective = best_objective
  # @show best_subobjective
  # # @show rewards
  # # @show weights
  return o >= 0.0
  # return best_objective >= 0.0
end

function _ossb_compute_feasible_v(instance::CombinatorialInstance{T}, θ::Dict{T, Float64}, t::Vector{Float64}) where T
  # Compute a v such that all constraints are respected, supposing that the input t satisfies the ∑i x_i / t_i ≤ Δx² constraints.
  # Two conditions: 
  # - the objective is positive: v r_\max - θ^T t ≥ 0 
  # - the point is a convex combination of integer solutions: A t ≤ v b
  arm_to_idx = Dict(arm => idx for (idx, arm) in enumerate(all_arm_indices(instance)))
  I = _ossb_confusing_arms(instance, θ)
  # I = all_arm_indices(instance)
  I_all = all_arm_indices(instance)

  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)

  # First condition gives a first estimation of v.
  θ⊤t = sum(θ[i] * t[arm_to_idx[i]] for i in I_all)
  v_min = θ⊤t / r_max

  # The second condition requires an explicit model to work with.
  m, _, vars = get_lp_formulation(instance, Dict(arm => 0.0 for arm in all_arm_indices(instance)))
  var_to_arm = Dict(values(vars) .=> keys(vars))

  # Compute the number of constraints.
  n_constraints = 0
  for (F, S) in list_of_constraint_types(m)
    # Ignore the case where this is an integrality/binary/bound constraint, we only need linear constraints.
    if S == MOI.Integer || S == MOI.ZeroOne || F == JuMP.VariableRef
      continue
    end

    n_constraints += num_constraints(m, F, S)
  end

  # Precompute A^T t and b.
  At = zeros(Float64, n_constraints)
  b = zeros(Float64, n_constraints)
  
  i = 1
  for (F, S) in list_of_constraint_types(m)
    # Ignore the case where this is an integrality/binary/bound constraint, we only need linear constraints.
    if S == MOI.Integer || S == MOI.ZeroOne || F == JuMP.VariableRef
      continue
    end

    for c in all_constraints(m, F, S)
      lhs = value(c, vr -> t[arm_to_idx[var_to_arm[vr]]])
      rhs = MOI.constant(moi_set(constraint_object(c)))

      At[i] = lhs
      b[i] = rhs
      i += 1
    end
  end

  # Check if this value satisfies the second constraint, and increase if need be.
  v = v_min
  if ! all(At .<= v .* b)
    vals = At ./ b
    vals[isinf.(vals)] .= -Inf # In case some b is zero.
    v = maximum(vals)
  end

  return v
end
