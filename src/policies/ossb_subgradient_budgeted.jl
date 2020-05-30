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
  n_iter = algo.n_iter

  # Determine an initial solution.
  t0 = ones(d)
  while !_ossb_is_solution_feasible(instance, θ, t0, algo.solve_all_budgets_at_once)
    t0 .*= 2
  end
  v0 = _ossb_compute_feasible_v(instance, θ, t0)
  x0 = [t0..., v0]

  # Use a subgradient method on the problem.
  prev_combs = Dict{Dict{T, Float64}, Vector{Vector{T}}}()
  all_combs = Set{Vector{T}}()
  algo__ = OSSBExactSmart()

  function _comb_subproblem(x::Vector{Float64})
    t_dict = Dict{T, Float64}(arm => ε * x[arm_to_idx[arm]] for arm in keys(θ))
    if ! (t_dict in keys(prev_combs))
      combs = _ossb_exact_cg_separation(instance, algo__, t_dict, θ, r_max)
      prev_combs[t_dict] = combs
      push!(all_combs, combs...)
    end
    return prev_combs[t_dict]
  end

  function objective(x::Vector{Float64})
    obj_best = x[end] * r_max
    obj_actual = sum(θ[arm] * x[arm_to_idx[arm]] for arm in keys(θ))
    return obj_best - obj_actual
  end

  function penalised_objective(x::Vector{Float64})
    obj = objective(x)
    if length(combs) > 0
      comb = combs[1]
      obj += λ * ε * sum((i in comb) ? 1 / x[i] : 0.0 for i in 1:d)
    end
    return obj
  end

  function subgradient(x::Vector{Float64})
    combs = _comb_subproblem(x)
    
    g = [-θ_vec..., r_max]
    if length(combs) > 0
      comb = combs[1]
      g_h = Float64[[(i in comb) ? -1 / x[i]^2 : 0.0 for i in 1:d]..., 0.0]
      g .+= (λ * ε) .* g_h
    end

    return g
  end

  function project(x::Vector{Float64})
    # Project the new iterate. Use all previously found combinatorial solutions to ensure feasibility sooner (this improves convergence).
    proj_m, _, proj_vars = get_lp_formulation(copy(instance), θ) # A t ≤ b
    @variable(proj_m, proj_v >= 0)
    @variable(proj_m, proj_inv_t[arm in keys(θ)] >= 0)

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
      if F == JuMP.VariableRef || F == MOI.SingleVariable
        if S != MOI.GreaterThan{Float64}
          for c in all_constraints(proj_m, F, S)
            JuMP.delete(proj_m, c)
          end
        end
        continue
      end

      # If there is a RHS (JuMP normalises everything so that the RHS has only
      # constants), multiply by v. Due to normalisation, this term goes to the
      # LHS, and must see its sign reversed.
      for c in all_constraints(proj_m, F, S)
        if ! iszero(normalized_rhs(c))
          new_coeff = - normalized_rhs(c)
          set_normalized_rhs(c, 0.0)
          set_normalized_coefficient(c, proj_v, new_coeff)
        end
      end
    end

    @constraint(proj_m, proj_v <= v_max) # Upper bound on v.
    for arm in keys(θ) # Lower bound on t.
      @constraint(proj_m, proj_vars[arm] >= t_min)
    end

    if length(prev_combs) > 0
      for arm in keys(θ) # Define the proj_inv_t variable (only used for the added constraints).
        @constraint(proj_m, [proj_vars[arm], proj_inv_t[arm], sqrt(2)] in RotatedSecondOrderCone())
      end

      for x in all_combs # Enfore the previously found gap constraints.
        Δx² = (r_max - sum(θ[arm] for arm in x))^2
        if Δx² >= Δmin ^ 2
          lhs = sum(proj_inv_t[arm] for arm in x if arm in I)
          c = @constraint(proj_m, lhs <= Δx²)
        end
      end
    end

    @objective(proj_m, Min, sum((proj_vars[arm] - x[arm_to_idx[arm]])^2 for arm in keys(θ)) + (proj_v - x[end])^2)

    optimize!(proj_m)

    return Float64[[value(proj_vars[arm]) for arm in keys(θ)]..., value(proj_v)]
  end

  # Solve the nonsmooth problem.
  info_callback = nothing
  if with_trace
    gl_details = GravesLaiDetails()
    info_callback = (k, f, x, g, f_best, x_best, t_iter) -> begin
      # if k % 200 == 0 # TODO: enabled by parameter.
      #   println(k)
      # end

      gl_details.n_iterations = k
      push!(gl_details.objectives, f)
      push!(gl_details.gl_bounds, objective(x))
      push!(gl_details.time_per_iteration, t_iter / 1_000_000)
    end
  end

  p = ProjectedConstrainedNonSmoothProblem(objective, subgradient, project, instance.n_arms + 1, Minimise)
  # x_val = NonsmoothOptim.solve(p, BundleMethod(Gurobi.Optimizer, η, 1.0, algo.n_iter), x0, info_callback=info_callback)
  x_val = NonsmoothOptim.solve(p, ProjectedSubgradientMethod(ConstantStepSize(η), algo.n_iter), x0, info_callback=info_callback)

  # Return the solution that has been found.
  res = GravesLaiResults(instance, Dict{T, Float64}(idx_to_arm[i] => x_val[i] for i in 1:d), objective(x_val))
  if with_trace
    return res, gl_details
  else
    return res
  end
end
