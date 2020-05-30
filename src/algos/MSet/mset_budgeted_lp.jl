solve(i::BudgetedMSetInstance, ::LinearProgramming; kwargs...) = budgeted_msets_lp(i; kwargs...)

function _budgeted_msets_lp_sub(i::BudgetedMSetInstance, solver)
  model = Model(solver)
  @variable(model, x[1:length(values(i))], Bin)
  @objective(model, Max, dot(x, values(i)))
  @constraint(model, sum(x) <= m(i))
  @constraint(model, c, dot(x, weights(i)) >= 0)

  set_silent(model)

  return model, x, c
end

function budgeted_msets_lp(i::BudgetedMSetInstance; solver=nothing, β::Int=budget(i))
  # Solve for all budgets at once, even though it is not really more efficient than looping outside this function.
  model, x, c = _budgeted_msets_lp_sub(i, solver)
  set_normalized_rhs(c, β)
  optimize!(model)

  V = Array{Float64, 3}(undef, m(i), dimension(i) + 1, budget(i) + 1)
  S = Dict{Tuple{Int, Int, Int}, Vector{Int}}()

  if termination_status(model) == MOI.OPTIMAL
    sol = findall(JuMP.value.(x) .>= 0.5)
    V[m(i), 0 + 1, β + 1] = objective_value(model)
    S[m(i), 0, β] = sol
  else
    V[m(i), 0 + 1, β + 1] = -Inf
    S[m(i), 0, β] = Int[-1]
  end

  return BudgetedMSetSolution(i, S[m(i), 0, β], V, S)
end

function budgeted_msets_lp_select(i::BudgetedMSetInstance, budgets; solver=nothing)
  model, x, c = _budgeted_msets_lp_sub(i, solver)

  V = Array{Float64, 3}(undef, m(i), dimension(i) + 1, budget(i) + 1)
  S = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
  for budget in budgets
    set_normalized_rhs(c, budget)
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
      sol = findall(JuMP.value.(x) .>= 0.5)
      V[m(i), 0 + 1, budget + 1] = objective_value(model)
      S[m(i), 0, budget] = sol
    else
      V[m(i), 0 + 1, budget + 1] = -Inf
      S[m(i), 0, budget] = Int[-1]
    end
  end

  return BudgetedMSetSolution(i, S[m(i), 0, maximum(budgets)], V, S)
end

function budgeted_msets_lp_all(i::BudgetedMSetInstance; solver=nothing, max_budget::Int=budget(i))
  # Solve for all budgets at once, even though it is not really more efficient than looping outside this function.
  return budgeted_msets_lp_select(i, 0:max_budget, solver=solver)
end