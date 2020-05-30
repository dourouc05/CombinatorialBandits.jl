module Kombinator
  # TODO: introduce a graph problem in the abstract hierarchy? Graph-related functions could be defined on it.
  # TODO: document the generic functions.
  # TODO: refactor the package further to remove the old st_prim()-like names? Only solve(::CombinatorialInstance, ::CombinatorialAlgorithm)::CombinatorialSolution should be used. At least, don't export all these names.
  # TODO: rename matching to "perfect matching" and introduce "imperfect matching".
  # TODO: add a note to rather use LightGraph's algorithms instead of these in this package for anything series. 
  # TODO: add a link to other packages like LightGraph for the corresponding algorithms, so that they seamlessly integrate with the others?
  # TODO: how to add a link to functions like budgeted_msets_lp_select, budgeted_msets_lp_all? They can be quite useful (but do not bring much in terms of performance)

  using LinearAlgebra
  using LightGraphs
  using JuMP

  import Base: values
  import LightGraphs: src, dst
  import JuMP: value

  import Hungarian
  import Munkres

  abstract type CombinatorialInstance end
  abstract type CombinatorialSolution end
  abstract type CombinatorialAlgorithm end
  
  # Define the algorithm types here, at the global level: some names might be shared by several algorithms. 
  # This is also the reason why there is no supplementary abstraction layer (e.g. a base type for all 
  # bipartite-matching algorithms).
  # First, generic-named algorithms (dynamic, greedy, etc.); then, the ones with a recognised name (think Bellman-Ford). 
  struct DynamicProgramming <: CombinatorialAlgorithm; end # When the DP algorithm has no more specific name exists, like Bellman-Ford
  struct GreedyAlgorithm <: CombinatorialAlgorithm; end
  struct LagrangianAlgorithm <: CombinatorialAlgorithm; end # Usually, for a crude approximation with no guarantee
  struct LagrangianRefinementAlgorithm <: CombinatorialAlgorithm; end # Usually, for an approximation term
  struct IteratedLagrangianRefinementAlgorithm <: CombinatorialAlgorithm; end # Usually, for a constant approximation ratio

  struct BellmanFordAlgorithm <: CombinatorialAlgorithm; end
  struct HungarianAlgorithm <: CombinatorialAlgorithm; end
  struct PrimAlgorithm <: CombinatorialAlgorithm; end

  # The case of linear programming is usually a bit more complex: in some cases, there may be several formulations. 
  # In that case, don't use the generic object here, but rather a problem-specific object that can specify the 
  # formulation to use. 
  abstract type AbstractLinearProgramming <: CombinatorialAlgorithm; end
  struct LinearProgramming <: AbstractLinearProgramming; end

  """
      approximation_ratio(::CombinatorialInstance, ::CombinatorialAlgorithm)

  Returns the approximation ratio for this algorithm when run on the input instance. When the problem is minimising 
  a function (e.g., finding the minimum-cost path), the ratio is defined as one constant ``r \\le 1.0``
  (ideally, the lowest) such that 

  ``f(x^\\star) \\leq f(x) \\leq r \\cdot f(x^\\star),``

  where ``f(x^\\star)`` is the cost of the optimum solution and ``f(x)`` the one of the returned solution. On the 
  contrary, for a maximisation problem, the definition is reversed (with ``r \\le 1.0``):

  ``r \\cdot f(x^\\star) \\geq f(x) \\geq f(x^\\star).``

  For an exact algorithm, the ratio is always ``1.0``, for both minimisation and maximisation problems. 

  The returned ratio might be constant, if the algorithm provides a constant ratio; if the ratio is not constant (i.e.
  instance-dependent), it may either be a worst-case value or a truly instance-dependent ratio. Depending on the 
  algorithm, this behaviour might be tuneable. 

  If the algorithm has no guarantee, it should return `NaN`. 
  """
  function approximation_ratio(::CombinatorialInstance, ::CombinatorialAlgorithm)
    return 1.0
  end

  """
      approximation_term(::CombinatorialInstance, ::CombinatorialAlgorithm)

  Returns the approximation term for this algorithm when run on the input instance. When the problem is minimising 
  a function (e.g., finding the minimum-cost path), the term is defined as one constant ``t \\ge 0.0`` 
  (ideally, the lowest) such that 

  ``f(x^\\star) \\leq f(x) \\leq f(x^\\star) + t,``

  where ``f(x^\\star)`` is the cost of the optimum solution and ``f(x)`` the one of the returned solution. On the 
  contrary, for a maximisation problem, the definition is reversed (with ``t \\ge 0.0``):

  ``f(x^\\star) \\leq f(x) \\leq f(x^\\star) - t.``

  For an exact algorithm, the term is always ``0.0``, for both minimisation and maximisation problems. 

  The returned term might be constant, if the algorithm provides a constant term; if the term is not constant (i.e.
  instance-dependent), it may either be a worst-case value or a truly instance-dependent term. Depending on the 
  algorithm, this behaviour might be tuneable. 

  If the algorithm has no guarantee, it should return `NaN`. 
  """
  function approximation_term(::CombinatorialInstance, ::CombinatorialAlgorithm)
    return 0.0
  end

  # Include the actual contents of the package.
  include("helpers.jl")

  include("BipartiteMatching/matching.jl")
  include("BipartiteMatching/matching_hungarian.jl")
  include("BipartiteMatching/matching_dp.jl")
  include("BipartiteMatching/matching_budgeted.jl")
  include("BipartiteMatching/matching_budgeted_dp.jl")
  include("BipartiteMatching/matching_budgeted_lagrangian.jl")

  include("ElementaryPath/ep.jl")
  include("ElementaryPath/ep_dp.jl")
  include("ElementaryPath/ep_budgeted.jl")
  include("ElementaryPath/ep_budgeted_dp.jl")

  include("MSet/mset.jl")
  include("MSet/mset_greedy.jl")
  include("MSet/mset_dp.jl")
  include("MSet/mset_lp.jl")
  include("MSet/mset_budgeted.jl")
  include("MSet/mset_budgeted_dp.jl")
  include("MSet/mset_budgeted_lp.jl")

  include("SpanningTree/st.jl")
  include("SpanningTree/st_prim.jl")
  include("SpanningTree/st_budgeted.jl")
  include("SpanningTree/st_budgeted_lagrangian.jl")

  # Export all symbols. Code copied from JuMP.
  const _EXCLUDE_SYMBOLS = [Symbol(@__MODULE__), :eval, :include]

  for sym in names(@__MODULE__, all=true)
    sym_string = string(sym)
    if sym in _EXCLUDE_SYMBOLS || startswith(sym_string, "_")
      continue
    end
    if !(Base.isidentifier(sym) || (startswith(sym_string, "@") && Base.isidentifier(sym_string[2:end])))
      continue
    end
    @eval export $sym
  end
end
