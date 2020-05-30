# For proximal methods: https://github.com/kul-forbes/ProximalOperators.jl

module NonsmoothOptim

using LinearAlgebra

import JuMP

export OptimisationSense, Minimise, Maximise,
       UnconstrainedNonSmoothProblem, ConstrainedNonSmoothProblem, ProjectedConstrainedNonSmoothProblem, 
       NonSmoothSolver, UnconstrainedNonSmoothSolver, ConstrainedNonSmoothSolver, ProjectedConstrainedNonSmoothSolver, 
       solve, 
       StepSizeRule, step, ConstantStepSize, ConstantStepLength, InverseStepSize, 
       SubgradientMethod, ProjectedSubgradientMethod, BundleMethod

## Define an optimisation problem.

@enum OptimisationSense Minimise Maximise

"""
    abstract type NonSmoothProblem

Represents a nonsmooth optimisation problem. Any implementation must provide the following properties: 

  * `f`: the function to optimise. It takes a a vector of `dimension` Float64 and generates a single Float64.
  * `g`: a function that returns any subgradient of `f`. It takes a vector of `dimension` Float64 and generates 
    a vector of `dimension` Float64. 
  * `dimension`: the dimension of the problem, i.e. the number of variables (an `Integer`).
  * `sense`: the `OptimisationSense` when processing `f` (i.e. minimise or maximise).

No automatic differentiation is performed in this package. Indeed, these techniques are usually not ensured to be 
correct for nonsmooth functions. https://arxiv.org/pdf/1809.08530.pdf
"""
abstract type NonSmoothProblem end

"""
    struct UnconstrainedNonSmoothProblem <: NonSmoothProblem

Represents an unconstrained nonsmooth optimisation problem. This type provides a barebones implementation of the 
`NonSmoothProblem` interface. 
"""
struct UnconstrainedNonSmoothProblem <: NonSmoothProblem
  f::Function
  g::Function
  dimension::Int
  sense::OptimisationSense
end

"""
    abstract type ConstrainedNonSmoothProblem <: NonSmoothProblem

Represents a constrained nonsmooth optimisation problem. It does not imply any new properties with respect to a 
`NonSmoothProblem`, because they might heavily depend on the kind of optimisation solver that is used
(indicator function, polytope, projection operator, proximale operator, etc.)
"""
abstract type ConstrainedNonSmoothProblem <: NonSmoothProblem end

"""
    struct ProjectedConstrainedNonSmoothProblem

Represents a constrained nonsmooth optimisation problem whose constraints are imposed through a projection operator. 
This structure provides one more member to the `NonSmoothProblem` interface: 

  * `project`: the projection operator. It takes a a vector of `dimension` Float64 and generates a vector of 
    `dimension` Float64. The returned point must be feasible according to the definition of the optimisation problem.

TODO: a stricter interface for projection operators? I.e. closed-form expressions or iterative solvers (they need parameters; maybe use a smooth optimisation package for this, with just constraints to be added, like `NLPModels`?). 
"""
struct ProjectedConstrainedNonSmoothProblem <: ConstrainedNonSmoothProblem
  f::Function
  g::Function
  project::Function
  dimension::Int
  sense::OptimisationSense
end

## Interface for a solver. 

abstract type NonSmoothSolver end # Also contains the solver parameters, like tolerance, number of iterations, etc.
abstract type UnconstrainedNonSmoothSolver <: NonSmoothSolver end
abstract type ConstrainedNonSmoothSolver <: NonSmoothSolver end
abstract type ProjectedConstrainedNonSmoothSolver <: ConstrainedNonSmoothSolver end

function solve
  # Input: 
  #   - A problem to solve.
  #   - An NSO method.
  #   - An initial iterate.
  # Keyword arguments: 
  #   - info_callback: (k, f, x, g, f_best, x_best, t_iter) -> nothing
  # Output: best solution found (Vector{Float64})
end

## Rules for step sizes, likely to be used by several solvers.

abstract type StepSizeRule end

function step
  # Input: 
  #   - Rule to apply. 
  #   - Number of iterations. 
  #   - Norm of the subgradient.
  # Output: step (Float64)
end

struct ConstantStepSize <: StepSizeRule
  step::Float64
end

function step(rule::ConstantStepSize, ::Int, ::Float64)
  return rule.step
end

struct ConstantStepLength <: StepSizeRule
  step::Float64
end

function step(rule::ConstantStepLength, ::Int, sg_norm::Float64)
  return rule.step / sg_norm
end

struct InverseStepSize <: StepSizeRule # Square summable, but not summable
  step::Float64
end

function step(rule::InverseStepSize, n_iter::Int, ::Float64)
  if n_iter <= 0
    return rule.step
  else
    return rule.step / n_iter
  end
end

# Actual code.
include("subgradient.jl")
include("bundle.jl")

end