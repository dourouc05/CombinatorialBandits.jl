struct BudgetedSpanningTreeInstance{T, U} <: CombinatorialInstance
  graph::AbstractGraph{T}
  rewards::Dict{Edge{T}, Float64}
  weights::Dict{Edge{T}, U}
  budget::U
end

function _budgeted_spanning_tree_compute_value(i::Union{SpanningTreeInstance{T}, BudgetedSpanningTreeInstance{T, U}}, tree::Vector{Edge{T}}) where {T, U}
  return sum(i.rewards[(e in keys(i.rewards)) ? e : reverse(e)] for e in tree)
end

abstract type BudgetedSpanningTreeSolution{T, U} <: CombinatorialSolution
  # instance::BudgetedSpanningTreeInstance{T, U}
  # tree::Vector{Edge{T}}
  # value::Float64 # TODO: remove me, only useful for Lagrangian.
end

struct SimpleBudgetedSpanningTreeSolution{T, U} <: BudgetedSpanningTreeSolution{T, U}
  instance::BudgetedSpanningTreeInstance{T, U}
  tree::Vector{Edge{T}}
  value::Float64

  function SimpleBudgetedSpanningTreeSolution(instance::BudgetedSpanningTreeInstance{T, U}, tree::Vector{Edge{T}}) where {T, U}
    if length(tree) > 0
      return new{T, U}(instance, tree, _budgeted_spanning_tree_compute_value(instance, tree))
    else
      return new{T, U}(instance, tree, -Inf)
    end
  end

  function SimpleBudgetedSpanningTreeSolution(instance::BudgetedSpanningTreeInstance{T, U}) where {T, U}
    # No feasible solution.
    return new{T, U}(instance, edgetype(instance.graph)[], -Inf)
  end
end
