# Based on https://www.tandfonline.com/doi/abs/10.1080/01966324.1987.10737221
# TODO: make a package out of this function. Kombinator? 

function _ossb_exact_convex_combination(instance::CombinatorialInstance{T}, t::Dict{T, Float64}) where T
  # Input: a combinatorial problem, a (continuous) solution to the OSSB problem, 
  # *scaled so that v=1*. This ensures that the point is within the polytope. 

  # The solution to return: a set of solutions (each of them being a set of arms), 
  # their weights in the convex combination.
  # This representation assumes that all vertices of the polytope are actual solutions. 
  points = Vector{Int}[]
  weights = Float64[]

  # Convert the current point t as a vector (this is required to perform linear algebra).
  arms = all_arm_indices(instance) # Assimilate an arm and its index.
  idx_arm = Dict(i => arms[i] for i in 1:length(arms))
  d = instance.n_arms
  x = zeros(Float64, d)
  for (arm, weight) in t
    x[idx_arm[arm]] = weight
  end

  # Counters for the algorithm. 
  point = t # What remains to decompose.
  δ = 1.0 # Weight still to assign.

  # Start the algorithm. 

  # Find the currently tight constraints to create the matrix G.
  _, tight_matrix = _ossb_find_tight_constraints(instance, x, idx_arm)

  while rank(tight_matrix) < d
    # Find a new vertex by decomposing (what's left of) x.
    y_bar = x
    while rank(tight_matrix) < d
      # Find a solution dir such that G dir = 0 with dir ≠ 0. 
      if length(tight_matrix) == 0 # Easier to compute than rank(G) == 0
        # No constraint to satisfy. Take an arbitrary "solution". This is a pathologically 
        # underdetermined system.
        # Normalise by dividing by the norm. 
        dir = ones(d) ./ sqrt(d)
      else
        # This system is underdetermined, so that the usual syntax A\b is not guaranteed to work 
        # (Julia will return a minimum-norm solution in the underdetermined case, which is very likely zero).
        dir = vec(nullspace(tight_matrix)[:, 1])

        # Also normalise this vector, in case the previous computation gave strange results. 
        n = norm(dir, 2)
        @assert n > 1.0e-6 # Just in case...
        dir ./= n
      end

      # Compute a point y = y_bar - α d that touches a face of the polyhedron (at least one more 
      # constraint is tight there). 
      α_val, y_bar = _ossb_touch_facet(copy(instance), y_bar, dir, idx_arm)

      # Update tight_matrix for this vertex-to-be (or already a vertex).
      _, tight_matrix = _ossb_find_tight_constraints(instance, y_bar, idx_arm)
    end

    # Retrieve that point from x to continue the decomposition. 
    dir = x - y_bar
    # @show dir
    γ_val, x = _ossb_touch_facet(copy(instance), x, dir, idx_arm)
    δ_scale = γ_val / (1 + γ_val)

    # This point y is a vertex that plays a role in the convex decomposition.
    push!(points, round.(Int, y_bar))
    push!(weights, δ_scale * δ)

    δ *= 1 - δ_scale

    # Update tight_matrix for the rest of the point to decompose.
    _, tight_matrix = _ossb_find_tight_constraints(instance, x, idx_arm)
  end

  # The last iterate x must be an extreme point, as the rank of the tight constraints is d.
  push!(points, round.(Int, x))
  push!(weights, δ)

  return points, weights
end

function _ossb_find_tight_constraints(instance::CombinatorialInstance{T}, x::Vector{Float64}, idx_arm::Dict{T, Int}) where T
  # Assumption: point within the polytope; the combinatorial instance has an LP formulation. 

  # Get the LP formulation, without objective function (_). 
  m, _, vars = get_lp_formulation(instance, Dict(arm => 0.0 for arm in all_arm_indices(instance)))

  # vars is a dictionary of arms to variables, but we will need the reverse operation. 
  var_to_arm = Dict(values(vars) .=> keys(vars))

  # Iterate over the constraints.
  tight_single = JuMP.ConstraintRef[]
  tight_linear = JuMP.ConstraintRef[]
  for (F, S) in list_of_constraint_types(m)
    # Ignore the case where this is an integrality constraint (it does not add bounds). Simple bounds are required! 
    if S == MOI.Integer
      continue
    end

    if S == MOI.ZeroOne
      @assert F == JuMP.VariableRef # TODO: is MOI.SingleVariable possible here? Other functions should not be allowed.
      for c in all_constraints(m, F, S)
        lhs = value(jump_function(constraint_object(c)), vr -> x[var_to_arm[vr]])
        if abs(lhs) <= 1.0e-6 || abs(lhs - 1.0) <= 1.0e-6
          push!(tight_single, c)
        end
      end
    else
      for c in all_constraints(m, F, S)
        # If the constraint is tight at t, keep it. 
        lhs = value(c, vr -> x[var_to_arm[vr]])
        rhs = MOI.constant(moi_set(constraint_object(c)))
        if abs(lhs - rhs) <= 1.0e-6
          push!(tight_linear, c)
        end
      end
    end
  end

  # Retrieve the coefficients for each constraint.
  var_to_idx = Dict(vars[arm] => idx for (arm, idx) in idx_arm) # Must be rebuild for each new JuMP model.

  tight_to_coefficients = Dict{JuMP.ConstraintRef, Vector{Float64}}()
  sizehint!(tight_to_coefficients, length(tight_single) + length(tight_linear))
  
  nvars = MOI.get(m, MOI.NumberOfVariables())

  for c in tight_single
    tight_to_coefficients[c] = spzeros(nvars)
    var = jump_function(constraint_object(c))
    tight_to_coefficients[c][var_to_idx[var]] = 1.0
  end
  for c in tight_linear
    tight_to_coefficients[c] = spzeros(nvars)
    for (var, coeff) in jump_function(constraint_object(c)).terms
      tight_to_coefficients[c][var_to_idx[var]] = coeff
    end
  end

  # The returned set may very well be empty, if the solution hit the bound on v. 
  tight_matrix = (length(tight_to_coefficients) > 0) ? vcat(collect(values(tight_to_coefficients))'...) : zeros(0, 0)
  return tight_to_coefficients, tight_matrix
end

function _ossb_touch_facet(instance::CombinatorialInstance{T}, x::Vector{Float64}, dir::Vector{Float64}, idx_arm::Dict{T, Int}) where T
  # Maximise α such that ȳ = y - 2 α dir ∈ polytope(instance), where y = x - dir. 
  # Return this optimum α and ȳ = y - 2 α⋆ dir. 
  d = length(idx_arm)

  # First, get the standard formulation, without objective function (_). 
  msub, _, varssub = get_lp_formulation(instance, Dict(arm => 0.0 for arm in all_arm_indices(instance)))
  var_to_idx = Dict(varssub[arm] => idx for (arm, idx) in idx_arm) 

  # Then, build a similar model for this specific problem.
  # TODO: could also do it by inspection of all constraints.
  m = Model(instance.solver.solver) # TODO: only for LP-based solvers. 
  # TODO: write a dedicated solver for this (highly simplified) problem? It could work by simply shortening an interval, iterating through the constraints.
  set_silent(m) # TODO: maybe allow to configure this?
  @variable(m, α)
  @objective(m, Max, α)

  # Iterate through the constraints and build the new ones. 
  y_bar = x .+ α .* dir

  for (F, S) in list_of_constraint_types(msub)
    # Ignore the case where this is an integrality constraint (it does not add bounds).
    if S == MOI.Integer
      continue
    end

    if S == MOI.ZeroOne
      @assert F == JuMP.VariableRef # TODO: is MOI.SingleVariable possible here? Other functions should not be allowed.
      for c in all_constraints(msub, F, S)
        var = jump_function(constraint_object(c)) # A simple variable reference.
        lhs = y_bar[var_to_idx[var]]
        @constraint(m, 0 <= lhs <= 1)
      end
    else
      for c in all_constraints(msub, F, S)
        lhs = AffExpr() + jump_function(constraint_object(c)).constant # The constant should be zero (only present in MOI set).
        for (var, coeff) in jump_function(constraint_object(c)).terms
          lhs += coeff * y_bar[var_to_idx[var]]
        end
        @constraint(m, lhs ∈ moi_set(constraint_object(c)))
      end
    end
  end

  # Solve the new problem. (It should be really easy, as there is only one variable.)
  optimize!(m)
  α_val = value(α)
  y_bar_val = value.(y_bar)

  return α_val, y_bar_val
end
