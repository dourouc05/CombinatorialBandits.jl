solve(i::MSetInstance, ::DynamicProgramming; kwargs...) = msets_dp(i; kwargs...)

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