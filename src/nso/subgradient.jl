struct SubgradientMethod <: UnconstrainedNonSmoothSolver
  step::StepSizeRule
  n_iter::Int 
end

function solve(p::UnconstrainedNonSmoothProblem, params::SubgradientMethod, x0::Vector{Float64}; kwargs...) 
  return _solve_sd(p, params, x0; kwargs...)
end

struct ProjectedSubgradientMethod <: ProjectedConstrainedNonSmoothSolver
  step::StepSizeRule
  n_iter::Int 
end

function solve(p::ProjectedConstrainedNonSmoothProblem, params::ProjectedSubgradientMethod, x0::Vector{Float64}; kwargs...) 
  return _solve_sd(p, params, x0; kwargs...)
end

function _solve_sd(p::NonSmoothProblem, params::NonSmoothSolver, x0::Vector{Float64}; info_callback::Union{Nothing, Function}=nothing)
  # Assumption for p' and params' types: 
  # - either (UnconstrainedNonSmoothProblem, SubgradientMethod) 
  # - or (ProjectedConstrainedNonSmoothProblem, ProjectedSubgradientMethod).
  @assert length(x0) == p.dimension

  x_best = copy(x0)
  if p isa ProjectedConstrainedNonSmoothProblem
    x0 = p.project(x0)
  end
  f_best = p.f(x0)

  dir = ((p.sense == Minimise) ? -1.0 : 1.0)::Float64

  x = copy(x0)
  for k in 1:params.n_iter
    t0 = time_ns()

    # Perform a gradient step.
    g = p.g(x)
    x += dir * step(params.step, k, norm(g)) * g

    # Projection, if need be.
    if p isa ProjectedConstrainedNonSmoothProblem
      x = p.project(x)
    end

    # Only take the solution if it improves the objective function. 
    f = p.f(x)
    if p.sense == Minimise
      if f < f_best
        f_best = f
        x_best = x
      end
    else
      if f > f_best
        f_best = f
        x_best = x
      end
    end

    if info_callback !== nothing
      info_callback(k, f, x, g, f_best, x_best, time_ns() - t0)
    end
  end

  return x_best
end