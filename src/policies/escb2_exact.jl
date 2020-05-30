# Uses the same optimiser as the base instance. Only works if that solver understands SOCP.
struct ESCB2Exact <: ESCB2OptimisationAlgorithm; end

function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, ::ESCB2Exact,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64},
                                    sqrtlinear_weight::Float64, ::Int;
                                    with_trace::Bool=false) where T
  # Get a LP formulation
  if ! has_lp_formulation(instance)
    error("The exact formulation for ESCB-2 relies on a LP formulation, which the solver associated to this instance cannot provide.")
  end

  m, obj, vars = get_lp_formulation(instance, linear)

  # Encode the square root as a geometric mean, i.e. a SOCP.
  #     t = sqrt(b^T x) = geomean(b^T x, 1)
  #     (2 t)² + (1 - b^T x)² <= (1 + b^T x)²
  #     (1 + b^T x, 1 - b^T x, 2 * t) in SOCP, following JuMP's conventions
  # Add this reformulation only once in the model to avoid JuMP complaining.
  if :CombinatorialESCB2 in keys(m.ext)
    dict = m.ext[:CombinatorialESCB2]
    confidence_bonus = dict[:confidence_bonus]
    confidence_bonus_linear = dict[:confidence_bonus_linear]
    eq = dict[:eq]
    cone = dict[:cone]

    for i in keys(sqrtlinear) # TODO: what about coefficients that become zero? May that happen? In that case, they won't necessarily be in sqrtlinear, and therefore not updated.
      # Due to normalisation, all variables are considered on the left-hand side,
      # hence the minus sign.
      set_normalized_coefficient(eq, vars[i], - sqrtlinear[i])
    end
  else
    @variable(m, confidence_bonus_linear >= 0) # b^T x
    @variable(m, confidence_bonus >= 0) # t

    @constraint(m, eq, confidence_bonus_linear == sqrtlinear_weight * sum(sqrtlinear[i] * vars[i] for i in keys(sqrtlinear)))
    @constraint(m, cone, [1 + confidence_bonus_linear, 1 - confidence_bonus_linear, 2 * confidence_bonus] in SecondOrderCone())

    m.ext[:CombinatorialESCB2] = Dict(:confidence_bonus_linear => confidence_bonus_linear, :confidence_bonus => confidence_bonus, :eq => eq, :cone => cone)
  end

  @objective(m, Max, obj + confidence_bonus)

  # Solve this program.
  set_silent(m)
  t0 = time_ns()
  optimize!(m)
  t1 = time_ns()

  if termination_status(m) != MOI.OPTIMAL
    error("The model was not solved correctly.")
  end

  # Retrieve the optimum value and, if needed, the trace of the solving time.
  arm_indices = all_arm_indices(instance)
  sol = T[i for i in arm_indices if value(vars[i]) > 0.5]

  if with_trace
    run_details = ESCB2Details()
    run_details.n_iterations = 1
    run_details.best_objective = value(obj + confidence_bonus)
    run_details.solver_time = (t1 - t0) / 1_000_000_000

    return sol, run_details
  else
    return sol
  end
end
