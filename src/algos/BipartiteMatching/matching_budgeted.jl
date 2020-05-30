struct BudgetedBipartiteMatchingInstance{T, U} <: CombinatorialInstance
  matching::BipartiteMatchingInstance{T}
  weight::Dict{Edge{T}, U}
  budget::U

  function BudgetedBipartiteMatchingInstance(graph::AbstractGraph{T}, reward::Dict{Edge{T}, Float64}, weight::Dict{Edge{T}, U}, budget::U) where {T, U}
    matching = BipartiteMatchingInstance(graph, reward)
    return new{T, U}(matching, weight, budget)
  end
end

function _budgeted_bipartite_matching_compute_value(i::BudgetedBipartiteMatchingInstance{T, U}, solution::Vector{Edge{T}}) where {T, U}
  return _budgeted_bipartite_matching_compute_value(i.matching, solution)
end

function _budgeted_bipartite_matching_compute_value(i::BipartiteMatchingInstance{T}, solution::Vector{Edge{T}}) where T
  return sum(i.reward[(e in keys(i.reward)) ? e : reverse(e)] for e in solution)
end

function _budgeted_bipartite_matching_compute_weight(i::BudgetedBipartiteMatchingInstance{T, U}, solution::Vector{Edge{T}}) where {T, U}
  return sum(i.weight[(e in keys(i.weight)) ? e : reverse(e)] for e in solution)
end

abstract type BudgetedBipartiteMatchingSolution{T, U} <: CombinatorialSolution
  # instance::BudgetedBipartiteMatchingInstance{T, U}
  # solution::Vector{Edge{T}}
  # value::Float64 # TODO: remove me, only useful for Lagrangian.
end
