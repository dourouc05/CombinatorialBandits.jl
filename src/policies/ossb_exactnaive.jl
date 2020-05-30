# Uses the same optimiser as the base instance. Only works if that solver
# understands lazy constraints and SOCP.
struct OSSBExactNaive <: OSSBOptimisationAlgorithm
  solver
end

function optimise_graves_lai(instance::CombinatorialInstance{T},
                       algo::OSSBExactNaive,
                       θ::Dict{T, Float64}; with_trace::Bool=true) where T
  # No need for an LP formulation! Just need an optimisation solver, given in the `OSSBExactNaive` object.
  t0 = time_ns()

  # TODO: make it an external function that can be overridden by combinatorial instances.
  # TODO: probably wrong when order has some importance, like in paths.
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
  
  # Precompute useful results.
  d = dimension(instance)
  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)
  Δx(x) = r_max - sum(θ[i] for i in x)
  I = _ossb_confusing_arms(instance, θ)
  # I = all_arm_indices(instance)

  # Complete formulation:
  #   min   ∑ η_x Δ_x
  #   s.t.  η_x ≥ 0                 ∀ x ∈ X
  #         t_i = ∑ η_x x_i         ∀ i ∈ {1, 2… d}
  #         ∑ x_i / t_i ≤ Δ_x^2     ∀ x ∈ X

  # Create the optimisation program.
  m = Model(algo.solver)

  @variable(m, η[x in solutions] >= 0)
  @variable(m, t[i in I] >= 0)
  @variable(m, inv_t[i in I] >= 0)

  @constraint(m, c_t[i in I],       t[i] == sum(η[x] for x in solutions if i in x)) # This line 
  # supposes that all arms are used in at least one solution, otherwise the line will fail
  # (reduction on an empty collection). This is a safe assumption.
  @constraint(m, c_inv_t[i in I],   [t[i], inv_t[i], sqrt(2)] in RotatedSecondOrderCone())

  # Add the gap constraints and build the objective. Ignore the optimum solutions.
  obj = AffExpr()
  for x in solutions
    Δx² = (Δx(x))^2
    if Δx² >= 1.0e-5 # TODO: \Delta_min? 
      # Always add this solution to the objective function. 
      add_to_expression!(obj, Δx(x), η[x]) # Performance-wise, better than repeated +=.

      # Generate a constraint only if it is not vacuous (i.e. not 0 <= Δx²).
      if length([i for i in x if i in I]) > 0
        @constraint(m, sum(inv_t[i] for i in x if i in I) <= Δx²)
      end
    end
  end
  @objective(m, Min, obj)

  # Imagine that the previous operations can be performed without exhausting memory... 
  set_silent(m)
  optimize!(m)

  if termination_status(m) != MOI.OPTIMAL
    @show termination_status(m)
    println(m)
    error(termination_status(m))
  end

  η_val = [value(η[x]) for x in solutions]
  res = GravesLaiResults(objective_value(m), solutions, η_val)
  if with_trace
    return res, GravesLaiDetails((time_ns() - t0) / 1_000_000, objective_value(m))
  else
    return res
  end
end
