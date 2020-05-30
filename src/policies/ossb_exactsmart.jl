# Uses the same optimiser as the base instance. Only works if that solver
# understands lazy constraints and SOCP.
mutable struct OSSBExactSmart <: OSSBOptimisationAlgorithm
  silent_solver::Bool
  instance

  # Separation subproblem (built once, when need be, hence mutability of this struct).
  sub_m
  sub_vars
  sub_r
  sub_z
end

OSSBExactSmart() = OSSBExactSmart(true, nothing, nothing, nothing, nothing, nothing)
OSSBExactSmart(silent_solver::Bool) = OSSBExactSmart(silent_solver, nothing, nothing, nothing, nothing, nothing)

function _ossb_exact_cg_separation(instance::CombinatorialInstance{T},
                                   algo::OSSBExactSmart,
                                   t::Dict{T, Float64},
                                   θ::Dict{T, Float64}, 
                                   r_max::Float64) where T
  if algo.instance === nothing
    algo.instance = instance
  elseif algo.instance != instance
    error("The OSSBExactSmart has already been used with another combinatorial instance.")
  end

  # If the separation subproblem has not yet been created, do it right now.
  if algo.sub_m === nothing
    # For a given t, solve the following (quadratic) separation problem:
    #    max_{x ∈ X} ∑_{i \in I} x_i / t_i - Δ_x^2
    # Return a solution x such that the objective function is positive (i.e. a
    # constraint that is violated), or nothing.
    m, _, vars = get_lp_formulation(copy(instance), t) # t is ignored, as it's only used
    # in the objective. It has all required arms, so that's enough.

    # Make a conic formulation for z == Δ_x^2 where Δ_x = r_max - θ^T x:
    #    z >= (θ^T x)^2    ⟺    1 >= (θ^T x)^2 / z
    # This is equivalent to:
    #    Δx == θ^T x
    #    (z + 1, z - 1, 2 * Δx) ∈ SOC
    @variable(m, Δx, lower_bound=0.0)
    @variable(m, z, lower_bound=0.0)
    @constraint(m, Δx == r_max - sum(θ[i] * vars[i] for i in keys(t)))
    @constraint(m, [1 + z, 1 - z, 2 * Δx] in SecondOrderCone())

    # Store the model in the OSSB object.
    algo.sub_m = m
    algo.sub_vars = vars
    algo.sub_r = Δx
    algo.sub_z = z
  end

  I = _ossb_confusing_arms(instance, θ)
  # I = all_arm_indices(instance)

  # Create the objective function.
  obj_linear = sum(algo.sub_vars[arm] / t[arm] for arm in keys(t))# if arm in I)
  obj_Δx² = algo.sub_z
  # obj_Δx² = r_max^2 - 2 * r_max * algo.sub_r + algo.sub_z # Δx² = (θ^T x^\star - θ^T x)^2
  # #                                                                \_ r_max _/   \ r /, with z ≥ r²
  @objective(algo.sub_m, Max, obj_linear - obj_Δx²)

  # println(algo.sub_m)
  # @show r_max

  optimize!(algo.sub_m)

  if termination_status(algo.sub_m) != MOI.OPTIMAL
    @show termination_status(algo.sub_m)
    println(algo.sub_m)
    error(termination_status(algo.sub_m))
  end

  # # println(algo.sub_m)
  # @show value(algo.sub_r)
  # @show value(algo.sub_z)
  # @show objective_value(algo.sub_m)
  # @show result_count(algo.sub_m)

  status = termination_status(algo.sub_m)
  return if status == MOI.OPTIMAL && objective_value(algo.sub_m) > 1.0e-5 # TODO: \Delta_min?
    Vector{T}[T[arm for arm in keys(t) if value(algo.sub_vars[arm], result=i) > 0.5] for i in 1:result_count(algo.sub_m)]
    # T[arm for arm in keys(t) if value(algo.sub_vars[arm]) > 0.5]
  else
    Vector{T}[]
  end
end

function optimise_graves_lai(instance::CombinatorialInstance{T},
                             algo::OSSBExactSmart,
                             θ::Dict{T, Float64}; with_trace::Bool=true) where T
  # Must be able to get a LP formulation.
  if ! has_lp_formulation(instance)
    # TODO: with the inexact algorithm, the instance's algorithm will no more be forced to be LP-based! But still, the convex decomposition step requires an LP formulation.
    error("The exact formulation for OSSB relies on a LP formulation, " *
          "which the solver associated to this instance cannot provide.")
  end

  # This function deals with copying the instance as many times as necessary. 
  # Called functions should not deal with these details.
  # TODO: rework the interface so that get_lp_formulation always returns a *new* formulation? Currently, these functions always return the same model...
  t0 = time_ns()
  
  # Precompute useful results.
  d = dimension(instance)
  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)

  arm_to_idx = Dict(arm => idx for (idx, arm) in enumerate(all_arm_indices(instance)))
  idx_to_arm = Dict(idx => arm for (idx, arm) in enumerate(all_arm_indices(instance)))
  
  I = _ossb_confusing_arms(instance, θ)

  # Reduced formulation:
  #   min   v (θ^T x^\star) - θ^T t
  #   s.t.  A t ≤ v b
  #         ∑ x_i / t_i ≤ Δ_x^2     ∀ x ∈ X    (constraint generation)
  # v could be removed if the polytope is described as A t = b, but that would require fiddling 
  # with the formulation quite a lot if that is not the case. 

  instance = copy(instance)
  m, _, vars = get_lp_formulation(instance, θ) # A t ≤ b
  @variable(m, v >= 0)
  @variable(m, 0 <= t[arm in keys(θ)] <= 100) # TODO: param for this upper bound, if it is really needed.
  @variable(m, 0 <= inv_t[arm in keys(θ)] <= 100) # TODO: param for this upper bound, if it is really needed.

  # From "A t ≤ b" to "A t - v b ≤ 0".
  # Multiply the right-hand side by v, for all constraints.
  for (F, S) in JuMP.list_of_constraint_types(m)
    # Remove integrality constraints. (In this case, F is a single variable.)
    # This could be replaced by relax_integrality() of newer JuMP versions, but it would be slower
    # (relax_integrality is reversible, which is not needed here).
    if S in [MOI.ZeroOne, S == MOI.Integer]
      for c in all_constraints(m, F, S)
        JuMP.delete(m, c)
      end
    end
    # Ignore the case where this is a simple bounds.
    if F == JuMP.VariableRef || F == MOI.SingleVariable
      continue
    end

    # If there is a RHS (JuMP normalises everything so that the RHS has only
    # constants), multiply by v. Due to normalisation, this term goes to the
    # LHS, and must see its sign reversed.
    for c in all_constraints(m, F, S)
      if ! iszero(normalized_rhs(c))
        coeff = normalized_rhs(c)
        set_normalized_rhs(c, 0.0)
        set_normalized_coefficient(c, v, -coeff)
      end
    end
  end

  # Define the inv_t variable (only used for the added constraints).
  for arm in keys(θ) # I
    @constraint(m, [vars[arm], inv_t[arm], sqrt(2)] in RotatedSecondOrderCone())
  end

  # Set the objective function.
  obj_best = v * r_max
  obj_actual = sum(θ[arm] * vars[arm] for arm in keys(θ))
  @objective(m, Min, obj_best - obj_actual)
  @constraint(m, obj_best - obj_actual >= 0) # Avoid numerical problems with some polytopes.

  if algo.silent_solver
    set_silent(m)
  else
    unset_silent(m)
  end

  # And go for constraint generation!
  prev_sols = Set{Vector{T}}() # Safety net. Should not be triggered (otherwise, the test on the objective value 
  # in _ossb_exact_cg_separation may need to be loosened).
  println("=> Starting constraint generation")
  while true
    optimize!(m)

    if termination_status(m) != MOI.OPTIMAL
      @show termination_status(m)
      println(m)
      error(termination_status(m))
    end

    # Use the separation oracle. Only care for values in I, and not all arms.
    # t_val = Dict{T, Float64}(arm => value(vars[arm]) for arm in I)
    t_val = Dict{T, Float64}(arm => value(vars[arm]) for arm in keys(θ))
    x_vecs = _ossb_exact_cg_separation(instance, algo, t_val, θ, r_max)
    
    if length(x_vecs) == 0 # No more constraint to add!
      break
    end

    added_constraint = false
    for x_vec in x_vecs
      if x_vec in prev_sols # || all(!(arm in I) for arm in x_vec)
        continue
      end
      push!(prev_sols, x_vec)

      # Found a violated constraint!
      Δx² = (r_max - sum(θ[arm] for arm in x_vec))^2
      if Δx² >= 1.0e-5 # TODO: rather use \Delta_min? 
        lhs = sum(inv_t[arm] for arm in x_vec if arm in I)
        # lhs = sum(inv_t[arm] for arm in x_vec)
        c = @constraint(m, lhs <= Δx²)
        added_constraint = true
      end
    end
    
    if ! added_constraint
      println(":: No constraint to add")
      break
    end

    println(":: Did one iteration of constraint generation")
  end
  println("=> Done with constraint generation")

  # Retrieve the solution and normalise with respect to v.
  # value. doesn't broadcast on dictionaries. 
  v_val = value(v)
  t_val = Dict{T, Float64}(arm => value(vars[arm]) / v_val for arm in keys(θ))

  # If the solution t_val is zero, there is not much to do. This should never happen, unless there is a serious
  # bug in the rest of the function.
  if iszero(collect(values(t_val)))
    println("Uh ho...")
    return solve_linear(instance, θ) # Perform pure exploitation.
  end

  println(m)

  # Return the solution that has been found.
  res = GravesLaiResults(instance, t_val, objective_value(m))
  if with_trace
    return res, GravesLaiDetails((time_ns() - t0) / 1_000_000, objective_value(m))
  else
    return res
  end
end
