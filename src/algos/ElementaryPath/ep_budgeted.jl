struct BudgetedElementaryPathInstance{T} <: CombinatorialInstance
  graph::AbstractGraph{T}
  rewards::Dict{Edge{T}, Float64}
  weights::Dict{Edge{T}, Int}
  src::T
  dst::T

  budget::Int # weights * solution >= budget

  function BudgetedElementaryPathInstance(graph::AbstractGraph{T}, rewards::Dict{Edge{T}, Float64},
                                          weights::Dict{Edge{T}, Int}, src::T, dst::T;
                                          budget::Union{Nothing, Int}=nothing,
                                          max_weight::Union{Nothing, Int}=nothing) where T
    # Error checking.
    if src == dst
      error("Source node is the same as destination node.")
    end

    if ! isnothing(budget) && budget < 0
      error("Budget is present and negative.")
    end

    if any(collect(values(weights)) .< 0)
      error("At least a weight is negative.")
    end

    if length(rewards) != length(weights)
      error("Not the same number of values and weights; these two vectors must have the same size.")
    end

    # Complete the optional parameters.
    if isnothing(max_weight) # Not useful to store, it is just used to compute the max budget if it is not given.
      max_weight = maximum(collect(values(weights)))
    end

    if isnothing(budget)
      d = length(values) # Dimension of the problem.
      budget = d * max_weight
    end

    # Return a new instance.
    new{T}(graph, rewards, weights, src, dst, budget)
  end
end

graph(i::BudgetedElementaryPathInstance{T}) where T = i.graph
rewards(i::BudgetedElementaryPathInstance{T}) where T = i.rewards
weights(i::BudgetedElementaryPathInstance{T}) where T = i.weights
src(i::BudgetedElementaryPathInstance{T}) where T = i.src
dst(i::BudgetedElementaryPathInstance{T}) where T = i.dst
budget(i::BudgetedElementaryPathInstance{T}) where T = i.budget

# dimension(i::BudgetedElementaryPathInstance{T}) where T = ne(graph(i))
reward(i::BudgetedElementaryPathInstance{T}, u::T, v::T) where T = rewards(i)[Edge(u, v)]
weight(i::BudgetedElementaryPathInstance{T}, u::T, v::T) where T = weights(i)[Edge(u, v)]

struct BudgetedElementaryPathSolution{T} <: CombinatorialSolution
  instance::BudgetedElementaryPathInstance{T}
  path::Vector{Edge{T}}
  states::Dict{Tuple{T, Int}, Float64}
  solutions::Dict{Tuple{T, Int}, Vector{Edge{T}}}
end

function paths_all_budgets(s::BudgetedElementaryPathSolution{T}, max_budget::Int) where T
  if max_budget > budget(s.instance)
    @warn "The asked maximum budget $max_budget is higher than the instance budget $(budget(s.instance)). Therefore, some values have not been computed and are unavailable."
  end

  mb = min(max_budget, budget(s.instance))
  return Dict{Int, Vector{Edge{T}}}(
    budget => s.solutions[s.instance.dst, budget] for budget in 0:mb)
end

function paths_all_budgets_as_tuples(s::BudgetedElementaryPathSolution{T}, max_budget::Int) where T
  if max_budget > budget(s.instance)
    @warn "The asked maximum budget $max_budget is higher than the instance budget $(budget(s.instance)). Therefore, some values have not been computed and are unavailable."
  end

  mb = min(max_budget, budget(s.instance))
  return Dict{Int, Vector{Tuple{T, T}}}(
    budget => [(src(e), dst(e)) for e in s.solutions[s.instance.dst, budget]]
    for budget in 0:mb)
end
