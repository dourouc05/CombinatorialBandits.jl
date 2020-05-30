abstract type OSSBSubgradientBudgetedParameterChoice end

struct OSSBSubgradientBudgetedAutomaticParameterChoice <: OSSBSubgradientBudgetedParameterChoice
  δ::Float64
  n_iter::Int
end

function _gl_subgradient_budgeted_compute_λ(p::OSSBSubgradientBudgetedAutomaticParameterChoice, ::CombinatorialInstance{T}, θ::Dict{T, Float64}) where T
  # Determine the dual multiplier. 
  f∞ = maximum(values(θ))
  λ = f∞ / p.δ
  return λ
end

function _gl_subgradient_budgeted_compute_η(p::OSSBSubgradientBudgetedAutomaticParameterChoice, instance::CombinatorialInstance{T}, θ::Dict{T, Float64}) where T
  # Determine the dual multiplier. 
  λ = _gl_subgradient_budgeted_compute_λ(p, instance, θ)

  # Precompute useful constants. 
  d = dimension(instance)
  m = maximum_solution_length(instance)
  Δmax = estimate_Δmax(instance, θ)
  Δmin = estimate_Δmin(instance, θ)

  # Determine the step.
  ε = approximation_ratio_budgeted(instance)
  t_max = d * log(m)^2 * sqrt(m) / Δmin^2 
  t_min = 1 / Δmax^2
  Δt = t_max - t_min
  Lf = norm(values(θ)) # Lipschitz constant of the (linear) objective function.
  η = (d * Δt^2) / (p.n_iter * (Lf^2 + (d * λ^2) / (ε^2 * t_min^2)))

  return η
end

struct OSSBSubgradientBudgetedFixedParameterChoice <: OSSBSubgradientBudgetedParameterChoice
  λ::Float64
  η::Float64
end

_gl_subgradient_budgeted_compute_λ(p::OSSBSubgradientBudgetedFixedParameterChoice, ::Any, ::Any) = p.λ
_gl_subgradient_budgeted_compute_η(p::OSSBSubgradientBudgetedFixedParameterChoice, ::Any, ::Any) = p.η

struct OSSBSubgradientBudgeted <: OSSBOptimisationAlgorithm
  params::OSSBSubgradientBudgetedParameterChoice
  n_iter::Int
  solve_all_budgets_at_once::Bool
end

# Constructor for automatic parameter choice.
OSSBSubgradientBudgeted(δ::Float64, n_iter::Int, solve_all_budgets_at_once::Bool) =
  OSSBSubgradientBudgeted(OSSBSubgradientBudgetedAutomaticParameterChoice(δ, n_iter), 
                          n_iter, solve_all_budgets_at_once)

# Constructor for fixed parameter choice.
OSSBSubgradientBudgeted(λ::Float64, η::Float64, n_iter::Int, solve_all_budgets_at_once::Bool) =
  OSSBSubgradientBudgeted(OSSBSubgradientBudgetedFixedParameterChoice(λ, η), 
                          n_iter, solve_all_budgets_at_once)

function optimise_graves_lai(instance::CombinatorialInstance{T},
                             algo::OSSBSubgradientBudgeted,
                             θ::Dict{T, Float64}; with_trace::Bool=true) where T
  @assert has_lp_formulation(instance) # TODO: implement in a more generic way (i.e. LP formulation is a property of the combinatorial problem, not of its solver).

  # Precompute useful constants for this instance. 
  ε = approximation_ratio_budgeted(instance)
  d = dimension(instance)

  m = maximum_solution_length(instance)
  Δmax = estimate_Δmax(instance, θ)
  Δmin = estimate_Δmin(instance, θ)

  v_max = d^3 / Δmin^2
  t_min = 1 / Δmax^2
  # t_max = d * log(m)^2 * sqrt(m) / Δmin^2 # Problematic when m = 1...
  
  # I = all_arm_indices(instance)
  I = _ossb_confusing_arms(instance, θ)
  I_all = all_arm_indices(instance) # Only for gap.
  arm_to_idx = Dict(arm => idx for (idx, arm) in enumerate(all_arm_indices(instance)))
  idx_to_arm = Dict(idx => arm for (idx, arm) in enumerate(all_arm_indices(instance)))
  θ_vec = [θ[idx_to_arm[i]] for i in 1:d]

  # Compute the optimum solution.
  x_max = solve_linear(copy(instance), θ)
  r_max = sum(θ[arm] for arm in x_max)
  
  # Determine the parameters. 
  λ = _gl_subgradient_budgeted_compute_λ(algo.params, instance, θ)
  η = _gl_subgradient_budgeted_compute_η(algo.params, instance, θ)

  # Determine an initial solution.
  t0 = ones(d)
  while !_ossb_is_solution_feasible(instance, θ, t0, algo.solve_all_budgets_at_once)
    t0 .*= 2
  end
  v0 = _ossb_compute_feasible_v(instance, θ, t0)
  x0 = [t0..., v0]

  algo__ = OSSBExactSmart()
  last_x_comb = nothing
  existing_project_comb_solutions = Set{Vector{T}}() # Only for projection operator.

  # Create the functions for the nonsmooth problem: objective function and constraints.
  f = (x::Vector{Float64}) -> begin
    # Split the variables into t (vector, one component per arm) and v (scalar).
    t = x[1:end-1]
    v = x[end]

    # Compute the combinatorial solution.
    t_dict = Dict(idx_to_arm[idx] => t_val[idx] for idx in 1:length(all_arm_indices(instance)))
    exact_oracle = _ossb_exact_cg_separation(instance, algo__, t_dict, θ, r_max)
    # rewards = Dict{T, Float64}(i => 1 / t[arm_to_idx[i]] for i in I_all)
    # weights = Dict{T, Int}(i => round(Int, θ[i] / Δmin, RoundUp) for i in I)
    # last_x_comb, _ = _maximise_nonlinear_through_budgeted(instance, rewards, weights, round(Int, Δmax / Δmin), 
    #                                                       (sol, budget) -> sum(rewards[i] for i in sol) - (budget * Δmin - r_max)^2,
    #                                                       algo.solve_all_budgets_at_once)

    # Check if any combinatorial solution violates the constraints. If there are more than one, pick one at random.
    ##### the most violating solution.
    last_x_comb_violated = length(exact_oracle) != 0
    last_x_comb = last_x_comb_violated ? exact_oracle[1] : nothing

    # Compute the objective. 
    f1 = v * r_max - dot(θ_vec, t) # Objective function
    f2 = 0.0
    if last_x_comb_violated # i.e. f2 > 0
      # Constraint function: f2 = [f2l - (r_max - f2q)^2]^+
      #                                  \____ Δx² ____/
      f2l = 0 
      r_x = 0
      for i in 1:d
        # if i in x_comb
        if i in I && i in last_x_comb
          r_x += 1 / t[i]
          f2l += θ[i]
        end
      end
      # Δx² = (r_max - sum(θ[arm] for arm in last_x_comb))^2
      # r_x = sum(1 / t[arm_to_idx[arm]] for arm in last_x_comb)
      # f2 = r_x - Δx²
      f2 = r_x - (r_max - f2l)^2
      @show f2
    end

    @show f1 + λ * f2
    return f1 + λ * f2
  end
  g = (x::Vector{Float64}) -> begin 
    x *= ε

    # Split the variables into t (vector, one component per arm) and v (scalar).
    t = x[1:end-1]
    v = x[end]

    # @assert last_x_comb !== nothing

    # # Compute the combinatorial solution if it is not available in the global variable.
    # t_dict = Dict(idx_to_arm[idx] => t_val[idx] for idx in 1:length(all_arm_indices(instance)))
    # exact_oracle = _ossb_exact_cg_separation(instance, algo__, t_dict, θ, r_max)
    # last_x_comb = exact_oracle[1]
    # # if last_x_comb === nothing
    # #   rewards = Dict{T, Float64}(i => 1 / t[arm_to_idx[i]] for i in I_all)
    # #   weights = Dict{T, Int}(i => round(Int, θ[i] / Δmin, RoundUp) for i in I)
    # #   last_x_comb, _ = _maximise_nonlinear_through_budgeted(instance, rewards, weights, round(Int, Δmax / Δmin), 
    # #                                                         (sol, budget) -> sum(rewards[i] for i in sol) - (budget * Δmin - r_max)^2,
    # #                                                         algo.solve_all_budgets_at_once)
    # # end

    # # Check whether the combinatorial solution yields a violated constraint.
    
    # Compute the gradient. 
    g1 = [-θ_vec..., r_max] # Objective function
    g2 = if last_x_comb !== nothing # Constraint function
      [(i in last_x_comb) ? -1.0 / t[i]^2 : 0.0 for i in 1:d]
    else
      zeros(d)
    end
    g2 = [g2..., 0.0] # Add the v variable.
    return g1 + λ * ε * g2
  end

  # Create the projection operator.
  proj_m, _, proj_vars = get_lp_formulation(copy(instance), θ) # A t ≤ b

  @variable(proj_m, 0 <= proj_v <= v_max)
  @variable(proj_m, 0 <= proj_inv_t[arm in keys(θ)] <= 100)

  obj_best = proj_v * r_max
  obj_actual = sum(θ[arm] * proj_vars[arm] for arm in keys(θ))
  @constraint(proj_m, obj_best - obj_actual >= 0)
  
  for (F, S) in list_of_constraint_types(proj_m) # From "A t ≤ b" to "A t ≤ v b": 
    # multiply the right-hand side by v, go to continuous problem.

    # Remove integrality/binary constraints. Bounds from binary constraints should 
    # be removed too (convex combination).
    if S == MOI.ZeroOne || S == MOI.Integer
      for c in all_constraints(proj_m, F, S)
        JuMP.delete(proj_m, c)
      end
      continue
    end
    # Ignore the case where this is a simple lower bound. 
    # Upper bounds are only allowed on proj_v. 
    if F == JuMP.VariableRef || F == MOI.SingleVariable
      if S != MOI.GreaterThan{Float64}
        for c in all_constraints(proj_m, F, S)
          if proj_v != jump_function(constraint_object(c))
            JuMP.delete(proj_m, c)
          end
        end
      end
      continue
    end

    # If there is a RHS (JuMP normalises everything so that the RHS has only
    # constants), multiply by v. Due to normalisation, this term goes to the
    # LHS, and must see its sign reversed.
    for c in all_constraints(proj_m, F, S)
      if ! iszero(normalized_rhs(c))
        coeff = normalized_rhs(c)
        set_normalized_rhs(c, 0.0)
        set_normalized_coefficient(c, proj_v, -coeff)
      end
    end
  end

  for arm in keys(θ) # Lower bound on t.
    @constraint(proj_m, proj_vars[arm] >= t_min)
  end

  # TODO: nonconvex shit.
  set_optimizer_attribute(proj_m, "NonConvex", 2)

  for arm in keys(θ) # Define the proj_inv_t variable (only used for the added constraints).
    @constraint(proj_m, proj_vars[arm] * proj_inv_t[arm] == 1.0)
    # @constraint(proj_m, [proj_vars[arm], proj_inv_t[arm], sqrt(2)] in RotatedSecondOrderCone())
  end

  project = (x::Vector{Float64}) -> begin
    t_val = x[1:end-1]
    v_val = x[end]
    
    # Compute the combinatorial solution if it is not available in the global variable.
    t = Dict(idx_to_arm[idx] => t_val[idx] for idx in 1:length(all_arm_indices(instance)))
    exact_oracle = _ossb_exact_cg_separation(instance, algo__, t, θ, r_max)
    exact_oracle = [x for x in exact_oracle if !(x in existing_project_comb_solutions)]
    # rewards = Dict{T, Float64}(i => 1 / t_val[arm_to_idx[i]] for i in I_all)
    # weights = Dict{T, Int}(i => round(Int, θ[i] / Δmin, RoundUp) for i in I)
    # last_x_comb, _ = _maximise_nonlinear_through_budgeted(instance, rewards, weights, round(Int, Δmax / Δmin), 
    #                                                       (sol, budget) -> sum(rewards[i] for i in sol) - (budget * Δmin - r_max)^2,
    #                                                       algo.solve_all_budgets_at_once)

    norm_t = sum((proj_vars[idx_to_arm[idx]] - t_val[idx]) ^ 2 for idx in 1:length(all_arm_indices(instance)))
    norm_v = (proj_v - v_val) ^ 2
    obj = norm_t + norm_v
    @objective(proj_m, Min, obj)
    # proj_vars_vec = [proj_vars[idx] for idx in 1:length(all_arm_indices(instance))]
    # @objective(proj_m, Min, proj_v * r_max - dot(θ_vec, proj_vars_vec))

    for x in exact_oracle
      if !(x in existing_project_comb_solutions)# && length([arm for arm in x if arm in I]) > 0
        Δx² = (r_max - sum(θ[arm] for arm in x))^2
        # @show Δx²
        if Δx² >= 1.0e-5
          lhs = sum(proj_inv_t[arm] for arm in x)# if arm in I)
          c = @constraint(proj_m, lhs <= Δx²)
        end

        push!(existing_project_comb_solutions, x)
      end
    end

    optimize!(proj_m)

    # println(proj_m)

    projected_x = [[value(proj_vars[arm]) for arm in keys(arm_to_idx)]..., value(proj_v)]
    # t_val = projected_x[1:end-1]
    # v_val = projected_x[end]

    # # If no more constraint had to be added at this stage, stop the process, but only *after* projection on the conic combinations!
    # if length(exact_oracle) == 0
    #   break
    # end

    # t = Dict(idx_to_arm[idx] => t_val[idx] for idx in 1:length(all_arm_indices(instance)))
    # exact_oracle = _ossb_exact_cg_separation(instance, algo__, t, θ, r_max)
    # @show exact_oracle
    # @show last_x_comb
    # for x in exact_oracle
    #   # @show x in existing_project_comb_solutions
    #   @show sum(1 / t_val[arm] for arm in x) - (sum(θ[arm] for arm in x) - r_max)^2
    # end
    # if last_x_comb !== nothing
    #   @show sum(1 / t_val[arm] for arm in last_x_comb) - (sum(θ[arm] for arm in last_x_comb) - r_max)^2
    # end
    # if length(morethings) > 0
    #   for x in values(morethings)
    #     if length(x) > 0 && x != [-1]
    #       @show x
    #       @show sum(1 / t_val[arm] for arm in x) - (sum(θ[arm] for arm in x) - r_max)^2
    #     end
    #   end
    # else
    #   @show morethings
    # end

    # @show projected_x
    # if added
    #   @show last_x_comb::Vector{T}
    #   Δx² = (r_max - sum(θ[arm] for arm in last_x_comb))^2
    #   @show sum(value(proj_inv_t[arm]) for arm in last_x_comb if arm in I) - Δx²
    #   if c !== nothing
    #     @show c
    #     @show value(c)
    #   end
    # end
    # println("-----------")

    # @show projected_x
    return projected_x
  end

  # Solve the nonsmooth problem.
  info_callback = nothing
  if with_trace
    gl_details = GravesLaiDetails()
    info_callback = (k, f, x, g, f_best, x_best, t_iter) -> begin
      if k % 200 == 0 # TODO: enabled by parameter.
        println(k)
      end

      # Split the variables into t (vector, one component per arm) and v (scalar).
      t = x[1:end-1]
      v = x[end]

      # Compute the value of the Graves-Lai bound, without the penalisation term.
      obj_best = v * r_max
      obj_actual = sum(θ[arm] * t[arm_to_idx[arm]] for arm in keys(θ))
      gl_val = obj_best - obj_actual

      gl_details.n_iterations = k
      push!(gl_details.objectives, f)
      push!(gl_details.gl_bounds, gl_val)
      push!(gl_details.time_per_iteration, t_iter / 1_000_000)
    end
  end

  p = ProjectedConstrainedNonSmoothProblem(f, g, project, instance.n_arms + 1, Minimise)
  x_val = NonsmoothOptim.solve(p, ProjectedSubgradientMethod(ConstantStepSize(η), algo.n_iter), x0, info_callback=info_callback)

  # println("============")
  println(f(x_val))
  println(x_val)
  println(proj_m)

  # Rework the solution to only include t (and not v) and to have a dictionary. 
  t_val = x_val[1:end-1] # Get rid of v.
  v_val = x_val[end]
  t_dict = Dict(idx_to_arm[i] => t_val[i] for i in 1:d)

  # Compute the true objective function, without penalisation. This should be equal to f if there is no violated constraint.
  obj_best = v_val * r_max
  obj_actual = sum(θ[arm] * t_val[arm_to_idx[arm]] for arm in keys(θ))
  obj = obj_best - obj_actual

  # Return the solution that has been found.
  res = GravesLaiResults(instance, t_dict, obj)
  if with_trace
    return res, gl_details
  else
    return res
  end
end
