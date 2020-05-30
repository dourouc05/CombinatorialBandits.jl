# Uses the same optimiser as the base instance. Only works if that solver
# understands SOCP.
mutable struct OSSBExact <: OSSBOptimisationAlgorithm
  silent_solver::Bool
end

OSSBExact() = OSSBExact(true)

function optimise_graves_lai(instance::CombinatorialInstance{T},
                             algo::OSSBExact,
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
    # I = all_arm_indices(instance)

  # TODO: make it an external function that can be overridden by combinatorial instances.
  # TODO: probably wrong when order has some importance, like in paths.
  # TODO: copied from OSSBExactNaive.
  # Generate a list of solutions. Probably highly inefficient, but it should work for any kind of combinatorial 
  # instance, as long as is_feasible and is_partially_acceptable are correctly implemented.
  # Technique: for each item, consider that it may be a part of a solution or not. Hence, build all combinations, 
  # stop one branch of computations as soon as a set of arms can no longer lead to a solution (something is already
  # violating a constraint). 
  # Initially: either take the first item or not. Maybe the first item cannot lead to a feasible solution.
  potential_solutions = Vector{T}[T[], T[first(all_arm_indices(instance))]]
  filter!(p -> is_partially_acceptable(instance, p), potential_solutions)

  solutions = Vector{T}[]
  for i in all_arm_indices(instance)[2:end]
    # Rebuild the set of potential solutions in this loop: for each existing element, try adding i or keeping the same thing.
    partial_solutions = copy(potential_solutions)
    deleteat!(potential_solutions, 1:length(potential_solutions)) # Empty the vector, so that it can be filled again.

    for p in partial_solutions
      # If this is already a feasible solution, put it aside.
      if length(p) > 0 && is_feasible(instance, p) && !(p in solutions)
        push!(solutions, p)
      end

      # The current potential solution is already known to be... a potential solution.
      push!(potential_solutions, p)

      # The other part of the recursion: try adding the new item i. 
      p2 = [p..., i]
      if is_feasible(instance, p2)
        push!(solutions, p2)
      end
      if is_partially_acceptable(instance, p2)
        push!(potential_solutions, p2)
      end
    end
  end

  potential_solutions = nothing # Free some memory.

  # Reduced formulation:
  #   min   v (θ^T x^\star) - θ^T t
  #   s.t.  A t ≤ v b
  #         ∑ x_i / t_i ≤ Δ_x^2     ∀ x ∈ X
  # v could be removed if the polytope is described as A t = b, but that would require fiddling 
  # with the formulation quite a lot if that is not the case. 

  # TODO: copied from OSSBExactSmart.
  instance = copy(instance)
  m, _, vars = get_lp_formulation(instance, θ) # A t ≤ b
  @variable(m, v >= 0)
  # @variable(m, inv_t[arm in I] >= 0)
  @variable(m, inv_t[arm in keys(θ)] >= 0) # TODO: param for this upper bound, if it is really needed.
  # @variable(m, 0 <= inv_t[arm in keys(θ)] <= 10_000) # TODO: param for this upper bound, if it is really needed.

  # TODO: copied from OSSBExactSmart.
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

  try
    set_optimizer_attribute(m, "NonConvex", 2)
  catch e
    set_optimizer_attribute(m, "CPX_OPTIMALITYTARGET_OPTIMALGLOBAL", 3)
  end

  # TODO: copied from OSSBExactSmart.
  # Define the inv_t variable (only used for the added constraints).
  for arm in keys(θ) # I
    # Encode the constraint inv_t[i] = 1 / t[i] = 1 / vars[i]. Relax it: 
    #     t[i] ≥ 1 / inv_t[i]
    # This corresponds to a rotated second-order cone: 
    #     (t[i], inv_t[i], √2) ∈ RSOC(3)
    # By definition, an RSOC is: 
    #     2 × t[i] × inv_t[i] ≥ (√2)² = 2.
    #     t[i] × inv_t[i] ≥ 1
    # @constraint(m, [inv_t[arm] + vars[arm], inv_t[arm] - vars[arm], 2] in SecondOrderCone())
    # @constraint(m, [vars[arm], inv_t[arm], sqrt(2)] in RotatedSecondOrderCone())
    @constraint(m, vars[arm] * inv_t[arm] == 1) # NON CONVEX THING!
  end

  # TODO: copied from OSSBExactSmart.
  # Set the objective function.
  obj_best = v * r_max
  obj_actual = sum(θ[arm] * vars[arm] for arm in keys(θ))
  @objective(m, Min, obj_best - obj_actual)
  @constraint(m, obj_best - obj_actual >= 0) # Avoid numerical problems with some polytopes.

  # TODO: copied from OSSBExactSmart.
  if algo.silent_solver
    set_silent(m)
  else
    unset_silent(m)
  end

  # Generate all gap constraints, even if that's a lof of things to feed the solver with.
  for x in solutions
    Δx² = (r_max - sum(θ[arm] for arm in x))^2
    @show Δx²
    
    if Δx² >= 1.0e-5
      c = @constraint(m, sum(inv_t[arm] for arm in x) <= Δx²)
      println(c)
    end
  end

  # Solve the (very large) formulation.
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    @show termination_status(m)
    println(m)
    error(termination_status(m))
  end

  @show value(v)
  for arm in keys(θ)
    @show value(vars[arm])
    @show value(inv_t[arm])
  end

  println(m)

  # TODO: copied from OSSBExactSmart.
  # Retrieve the solution and normalise with respect to v.
  # value. doesn't broadcast on dictionaries. 
  v_val = value(v)
  t_val = Dict{T, Float64}(arm => value(vars[arm]) / v_val for arm in keys(θ))

  # TODO: copied from OSSBExactSmart.
  # If the solution t_val is zero, there is not much to do. This should never happen, unless there is a serious
  # bug in the rest of the function.
  if iszero(collect(values(t_val)))
    println("Uh ho...")
    return solve_linear(instance, θ) # Perform pure exploitation.
  end

  # TODO: copied from OSSBExactSmart.
  # Return the solution that has been found.
  res = GravesLaiResults(instance, t_val, objective_value(m))
  if with_trace
    return res, GravesLaiDetails((time_ns() - t0) / 1_000_000, objective_value(m))
  else
    return res
  end
end
