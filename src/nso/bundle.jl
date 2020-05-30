struct BundleMethod <: UnconstrainedNonSmoothSolver
  solver_factory # Must support QPs with linear constraints.
  µ::Float64 # Used in the proximal step. Considered to be constant for now. TODO: schedules
  K::Float64
  n_iter::Int
end

function solve(p::UnconstrainedNonSmoothProblem, params::BundleMethod, x0::Vector{Float64}; info_callback::Union{Nothing, Function}=nothing) 
  @assert p.sense == Minimise # TODO: current limitation.
  @assert length(x0) == p.dimension

  # Build the bundle model.
  m = JuMP.Model(params.solver_factory)
  f = @JuMP.variable(m)
  bx = @JuMP.variable(m, [1:p.dimension])

  # Create the first cutting plane with the starting iterate (i.e. the program will never be unbounded).
  @JuMP.constraint(m, f >= p.f(x0) + dot(p.g(x0), bx - x0))

  # Start iterating.
  x = copy(x0)
  for k in 1:params.n_iter
    t0 = time_ns()

    # Get the new test point (proximal step).
    @JuMP.objective(m, Min, f + params.µ * sum((bx[i] - x[i])^2 for i in 1:3))
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == MOI.OPTIMAL

    y = value.(bx)
    @JuMP.constraint(m, f >= p.f(y) + dot(p.g(y), bx - y))

    # Is this point optimum? 
    @JuMP.objective(m, Min, f)
    c = @JuMP.constraint(m, bx .== y)
    JuMP.optimize!(m)
    @assert JuMP.termination_status(m) == MOI.OPTIMAL

    f_bundle_y = JuMP.value(f)
    JuMP.delete(m, c)
    
    f = p.f(x)
    v = f_bundle_y - f
    if v >= 0
      break
    end

    # Perform a step: 
    if p.f(y) <= f + params.K * v
      # Descent step. 
      x = y
    else
      # Short step.
      nothing
    end

    if info_callback !== nothing
      info_callback(k, f, x, g, f_best, x_best, time_ns() - t0)
    end
  end

  return x
end
