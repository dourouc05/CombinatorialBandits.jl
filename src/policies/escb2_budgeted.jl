# Indicates how coefficients should be discretised for the budgeted implementation.
abstract type ESCB2BudgetedDiscretisationScheme end

struct ESCB2BudgetedDiscretisationSchemeConstant <: ESCB2BudgetedDiscretisationScheme
  v::Float64
end

struct ESCB2BudgetedDiscretisationSchemeLambda <: ESCB2BudgetedDiscretisationScheme
  f::Function # Argument: the current round t.
end

function _discretisation_provably_converging(instance::CombinatorialInstance, round::Int)
  logm = max(1.0, log(instance.m))
  δ = instance.n_arms * (logm ^ 2) / round
  return δ / instance.m
end

ESCB2BudgetedDiscretisationSchemeConservative(instance::CombinatorialInstance, max_rounds::Int) =
  ESCB2BudgetedDiscretisationSchemeConstant(_discretisation_provably_converging(instance, max_rounds))
ESCB2BudgetedDiscretisationSchemeAdaptive(instance::CombinatorialInstance) =
  ESCB2BudgetedDiscretisationSchemeLambda(t -> _discretisation_provably_converging(instance, t))

_get_ξ(::ESCB2BudgetedDiscretisationScheme, ::Int) = error("Not implemented")
_get_ξ(s::ESCB2BudgetedDiscretisationSchemeConstant, ::Int) = s.v
_get_ξ(s::ESCB2BudgetedDiscretisationSchemeLambda, t::Int) = s.f(t)

# Parameters for the budgeted algorithm.
struct ESCB2Budgeted <: ESCB2OptimisationAlgorithm
  discretisation_scheme::ESCB2BudgetedDiscretisationScheme
  solve_all_budgets_at_once::Union{Bool, Nothing} # Some optimisation algorithms can solve the budgeted problems for all values of the budget at once more efficiently.
end

ESCB2Budgeted(i::CombinatorialInstance, s::Union{Bool, Nothing}=nothing) =
  ESCB2Budgeted(ESCB2BudgetedDiscretisationSchemeAdaptive(i), s)
ESCB2Budgeted(ε::Float64, s::Union{Bool, Nothing}=nothing) =
  ESCB2Budgeted(ESCB2BudgetedDiscretisationSchemeConstant(ε), s)

# Optimisation of ESCB2's objective function.
function optimise_linear_sqrtlinear(instance::CombinatorialInstance{T}, algo::ESCB2Budgeted,
                                    linear::Dict{T, Float64}, sqrtlinear::Dict{T, Float64},
                                    sqrtlinear_weight::Float64, bandit_round::Int;
                                    with_trace::Bool=false) where T
  # Transform the linear term in a budget constraint, the nonlinear term
  # becoming the objective function (in which case the concave function
  # can be dropped).
  # For this to work, the linear part of the objective function must
  # take only integer values, i.e. the linear coefficients are actually
  # only integers.
  ξ = _get_ξ(algo.discretisation_scheme, bandit_round)
  linear_discrete = Dict(k => round(Int, v / ξ, RoundUp) for (k, v) in linear)

  # Fill automatic values.
  if algo.solve_all_budgets_at_once === nothing
    algo = copy(algo) # Don't modify the user's object.
    algo.solve_all_budgets_at_once = supports_solve_all_budgeted_linear(instance)
  end

  # Start timing.
  t0 = time_ns()

  # Maximum value of the linear term?
  max_budget = instance.n_arms * maximum(values(linear_discrete))

  # Solve the family of problems with an increasing budget.
  best_solution, best_objective = _maximise_nonlinear_through_budgeted(instance, sqrtlinear, linear_discrete, max_budget,
                                                                       (sol, _) -> escb2_index(linear, sqrtlinear, sqrtlinear_weight, sol),
                                                                       algo.solve_all_budgets_at_once)
  t1 = time_ns()

  # Return the nice data structures.
  if with_trace
    run_details = ESCB2Details()
    run_details.n_iterations = 1
    run_details.best_objective = best_objective
    run_details.solver_time = (t1 - t0) / 1_000_000_000
    return best_solution, run_details
  else
    return best_solution
  end
end
