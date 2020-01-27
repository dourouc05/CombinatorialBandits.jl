using CombinatorialBandits
using Test

using LightGraphs
using Random
using Distributions
using JuMP
# using Cbc # , SCS, Pajarito
using CPLEX, Gurobi; const Cbc = CPLEX # Gurobi for lazy constraints (does not seem to work with CPLEX)

function test_policy_trace(instance::CombinatorialInstance{T}, policy::Policy,
                           n_rounds::Int, total_arm_count::Int,
                           final_regret::Float64, final_reward::Float64,
                           expected_sum::Float64) where T
  Random.seed!(1)
  s, t = simulate(instance, policy, n_rounds, with_trace=true)

  # State should still be consistent.
  @test s.round == n_rounds
  @test sum(values(s.arm_counts)) <= total_arm_count
  @test s.regret ≈ final_regret atol=1.e-9
  @test s.reward ≈ final_reward
  @test s.regret + s.reward ≈ expected_sum

  # Now, the trace.
  @test length(t.states) == n_rounds
  @test length(t.arms) == n_rounds
  @test length(t.reward) == n_rounds
  @test length(t.policy_details) == n_rounds

  @test t.states[end].regret ≈ s.regret
  @test t.states[end].reward ≈ s.reward

  for i in 1:n_rounds
    @test t.states[i].round == i
    if i > 1
      @test t.states[i].reward >= t.states[i - 1].reward

      for arm in keys(t.states[i].arm_counts)
        @test t.states[i].arm_counts[arm] >= t.states[i - 1].arm_counts[arm]
        @test t.states[i].arm_reward[arm] >= t.states[i - 1].arm_reward[arm]
      end

      @test is_feasible(instance, t.arms[i])
    end

    for arm in keys(t.states[i].arm_counts)
      if t.states[i].arm_counts[arm] > 0
        @test t.states[i].arm_average_reward[arm] ≈ t.states[i].arm_reward[arm] / t.states[i].arm_counts[arm]
      end
    end
  end
end

import CombinatorialBandits: solve_linear

struct PerfectBipartiteMatchingNoSolver <: PerfectBipartiteMatchingSolver end
solve_linear(::PerfectBipartiteMatchingNoSolver, ::Dict{Tuple{Int64, Int64}, Float64}) = Tuple{Int, Int}[(1, 1)]

struct ElementaryPathNoSolver <: ElementaryPathSolver end
solve_linear(::ElementaryPathNoSolver, ::Dict{Tuple{Int64, Int64}, Float64}) = Tuple{Int, Int}[(1, 2)]

struct SpanningTreeNoSolver <: SpanningTreeSolver end
solve_linear(::SpanningTreeNoSolver, ::Dict{Tuple{Int64, Int64}, Float64}) = Tuple{Int, Int}[(1, 2)]

@testset "CombinatorialBandits.jl" begin
  @testset "Instances" begin
    include("instances_elementarypath.jl")
    include("instances_matching.jl")
    include("instances_spanningtree.jl")
  end

  @testset "Policies" begin
    include("policies_ts.jl")
    include("policies_llr.jl")
    include("policies_cucb.jl")
    include("policies_escb2.jl")
  end

  @testset "Combinatorial algorithms" begin
    include("algos/ep.jl")
    include("algos/msets.jl")
    include("algos/st.jl")
  end

  include("main.jl")
end
