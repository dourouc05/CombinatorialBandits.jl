function _solution_symmetric_difference(a::Vector{T}, b::Vector{T}) where T
  only_in_a = T[]
  only_in_b = T[]

  for e in a
    if e in b
      continue
    end
    push!(only_in_a, e)
  end

  for e in b
    if e in a
      continue
    end
    push!(only_in_b, e)
  end

  return only_in_a, only_in_b
end

function _solution_symmetric_difference_size(a::Vector{T}, b::Vector{T}) where T
  only_in_a, only_in_b = _solution_symmetric_difference(a, b)
  return length(only_in_a) + length(only_in_b)
end
