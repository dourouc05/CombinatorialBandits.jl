"""
An instance of the m-set problem.

Values are not restricted.

It can be formalised as follows:

``\\max \\sum_i \\mathrm{values}_i x_i``
``\\mathrm{s.t.} \\sum_i x_i \\leq m, \\quad x \\in \\{0, 1\\}^d``
"""
struct MSetInstance <: CombinatorialInstance
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

struct MSetSolution <: CombinatorialSolution
  instance::MSetInstance
  items::Vector{Int} # Indices to the chosen items.
end

function value(s::MSetSolution)
  return sum(s.instance.values[i] for i in s.items)
end
