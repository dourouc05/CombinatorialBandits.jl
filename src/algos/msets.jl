import Base: values
using LinearAlgebra

"""
An instance of the m-set problem.

Values are not restricted.

It can be formalised as follows:

``\\max \\sum_i \\mathrm{values}_i x_i``
``\\mathrm{s.t.} \\sum_i x_i \\leq m, \\quad x \\in \\{0, 1\\}^d``
"""
struct MSetInstance
  values::Vector{Float64}
  m::Int

  function MSetInstance(values::Vector{Float64}, m::Int)
    # Error checking.
    if m < 0
      error("m is less than zero: there is no solution.")
    end

    if m == 0
      error("m is zero: the only solution is to take no items.")
    end

    # Return a new instance.
    new(values, m)
  end
end

values(i::MSetInstance) = i.values
m(i::MSetInstance) = i.m

value(i::MSetInstance, o::Int) = values(i)[o]
values(i::MSetInstance, o) = values(i)[o]
dimension(i::MSetInstance) = length(values(i))

struct MSetSolution
  instance::MSetInstance
  items::Vector{Int} # Indices to the chosen items.
end

function value(s::MSetSolution)
  return sum(s.instance.values[i] for i in s.items)
end

function msets_greedy(instance::MSetInstance)
  # Algorithm: sort the weights, take the m largest ones, this is the optimum m-set solution.
  # Implementation: no need for sorting, partialsortperm returns the largest items.
  items = collect(partialsortperm(instance.values, 1:instance.m, rev=true))
  return MSetSolution(instance, items)
end

function msets_dp(i::MSetInstance)
  # V[µ, δ]: µ items, up to δ.
  V = Matrix{Float64}(undef, m(i), dimension(i))
  S = Dict{Tuple{Int, Int}, Vector{Int}}()

  # Initialise: µ == 1, just take the best element among [δ+1, d]; δ == d, take nothing.
  V[1, dimension(i)] = 0.0
  S[1, dimension(i)] = Int[]

  for δ in (dimension(i) - 1):-1:1
     v, x = findmax(values(i, (δ + 1):dimension(i)))
     V[1, δ] = v
     S[1, δ] = [x]
  end

  for µ in 2:m(i)
    V[µ, dimension(i)] = 0.0
    S[µ, dimension(i)] = Int[]
  end

  # Dynamic part.
  for µ in 2:m(i)
    for δ in (dimension(i) - 1):-1:0
      obj_idx = δ + 1
      take_δ = value(i, obj_idx) + V[µ - 1, δ + 1]
      dont_take_δ = V[µ, δ + 1]

      if take_δ > dont_take_δ
        V[µ, δ + 1] = take_δ
        S[µ, δ + 1] = vcat(δ, S[µ - 1, δ + 1])
      else
        V[µ, δ + 1] = dont_take_δ
        S[µ, δ + 1] = S[µ, δ + 1]
      end
    end
  end

  return MSetSolution(i, S[m(i), 1] .+ 1)
end

function msets_lp(i::MSetInstance; solver=nothing)
  m = Model(solver)
  @variable(m, x[1:length(i.values)], Bin)
  @objective(m, Max, dot(x, i.values))
  @constraint(m, sum(x) <= i.m)

  set_silent(m)
  optimize!(m)

  return MSetSolution(i, findall(JuMP.value.(x) .>= 0.5))
end
