@testset "Perfect bipartite matching" begin
  @testset "Uncorrelated" begin
    @testset "Constructor with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      @test instance.n_arms == n ^ 2
      @test instance.reward == reward

      # Error: non bipartite.
      reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:(n + 2)]
      @test_throws ErrorException UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:(n + 2), j in 1:n]
      @test_throws ErrorException UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())
    end

    @testset "State with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : .0)) for i in 1:n, j in 1:n]
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      state = initial_state(instance)

      @test state.round == 0
      @test state.regret == 0.0
      @test state.reward == 0.0
      @test length(state.arm_counts) == n * n
      @test length(state.arm_reward) == n * n
      @test length(state.arm_average_reward) == n * n

      for i in 1:n
        for j in 1:n
          @test state.arm_counts[(i, j)] == 0
          @test state.arm_reward[(i, j)] == 0.0
          @test state.arm_average_reward[(i, j)] == 0.0
        end
      end
    end

    @testset "Trace with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : .0)) for i in 1:n, j in 1:n]
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      trace = initial_trace(instance)

      @test length(trace.states) == 0
      @test length(trace.arms) == 0
      @test length(trace.reward) == 0
      @test length(trace.policy_details) == 0
      @test length(trace.time_choose_action) == 0

      @test eltype(trace.states) == State{Tuple{Int, Int}}
      @test eltype(trace.arms) == Vector{Tuple{Int, Int}}
      @test eltype(trace.reward) == Vector{Float64}
      @test eltype(trace.time_choose_action) == Int
    end

    @testset "Pull with $i nodes on each side" for i in [2, 5, 10]
      n = i
      reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      Random.seed!(1)
      @test pull(instance, [(1, 2), (i, i)]) == ([1.0, 0.0], -1) # Reward and regret
    end

    @testset "Check feasibility with 3 nodes on each side" begin
      n = 3
      reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      @test is_feasible(instance, [(1, 2), (2, 3), (3, 1)])
      @test ! is_feasible(instance, [(2, 3), (3, 2), (1, 2)])
      @test ! is_feasible(instance, [(2, 3), (3, 2), (2, 1), (1, 2)])
    end

    if ! is_travis
      @testset "LP solver" begin
        @testset "Constructor" for i in [2, 5, 10]
          n = i
          reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
          instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))

          @test instance.solver != nothing
          @test instance.solver.model != nothing
          @test size(instance.solver.x, 1) == n * n
        end

        @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
          n = i
          reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
          instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))

          Random.seed!(i)
          drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
          solution = solve_linear(instance, drawn)
          @test is_feasible(instance, solution)
        end
      end
    end

    @testset "Munkres solver" begin
      @testset "Constructor" for i in [2, 5, 10]
        n = i
        reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
        instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

        @test instance.solver != nothing
      end

      @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
        n = i
        reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
        instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

        Random.seed!(i)
        drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
        solution = solve_linear(instance, drawn)
        @test is_feasible(instance, solution)
      end
    end

    @testset "Hungarian solver" begin
      @testset "Constructor" for i in [2, 5, 10]
        n = i
        reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
        instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

        @test instance.solver != nothing
      end

      @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
        n = i
        reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
        instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

        Random.seed!(i)
        drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
        solution = solve_linear(instance, drawn)
        @test is_feasible(instance, solution)
      end
    end

    @testset "Solver equivalence (size: $i nodes on each side)" for i in [2, 5, 10]
      n = i
      reward = Distribution[Bernoulli(((i == j) ? 0.0 : 1.0)) for i in 1:n, j in 1:n]
      instance_lp = ! is_travis && UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))
      instance_munkres = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())
      instance_hungarian = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

      Random.seed!(i)
      drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)

      if ! is_travis
        solution_lp = solve_linear(instance_lp, drawn)
        @test is_feasible(instance_lp, solution_lp)
      end

      solution_munkres = solve_linear(instance_munkres, drawn)
      @test is_feasible(instance_munkres, solution_munkres)

      solution_hungarian = solve_linear(instance_hungarian, drawn)
      @test is_feasible(instance_hungarian, solution_hungarian)

      # All solutions must have the same length, as these are perfect matchings.
      @test length(solution_hungarian) == length(solution_munkres)

      cost_munkres = sum(drawn[o] for o in solution_munkres)
      cost_hungarian = sum(drawn[o] for o in solution_hungarian)

      @test cost_hungarian ≈ cost_munkres
      if ! is_travis
        @test length(solution_lp) == length(solution_hungarian)
        cost_lp = sum(drawn[o] for o in solution_lp)
        @test cost_lp ≈ cost_hungarian
      end
    end
  end

  @testset "Correlated" begin
    @testset "Constructor with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      μ = vec(Float64[.5 + ((i == j) ? ε : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      @test instance.n_arms == n ^ 2
      @test instance.reward == reward
    end

    @testset "State with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      μ = vec(Float64[.5 + ((i == j) ? ε : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      state = initial_state(instance)

      @test state.round == 0
      @test state.regret == 0.0
      @test state.reward == 0.0
      @test length(state.arm_counts) == n * n
      @test length(state.arm_reward) == n * n
      @test length(state.arm_average_reward) == n * n

      for i in 1:n
        for j in 1:n
          @test state.arm_counts[(i, j)] == 0
          @test state.arm_reward[(i, j)] == 0.0
          @test state.arm_average_reward[(i, j)] == 0.0
        end
      end
    end

    @testset "Trace with $i nodes on each side" for i in [2, 5, 10]
      n = i
      ε = (i - 2) / 16
      μ = vec(Float64[.5 + ((i == j) ? ε : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      trace = initial_trace(instance)

      @test length(trace.states) == 0
      @test length(trace.arms) == 0
      @test length(trace.reward) == 0
      @test length(trace.policy_details) == 0
      @test length(trace.time_choose_action) == 0

      @test eltype(trace.states) == State{Tuple{Int, Int}}
      @test eltype(trace.arms) == Vector{Tuple{Int, Int}}
      @test eltype(trace.reward) == Vector{Float64}
      @test eltype(trace.time_choose_action) == Int
    end

    @testset "Pull with $i nodes on each side" for i in [2]#, 5, 10]
      n = i
      μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      Random.seed!(1)
      rewards, regret = pull(instance, [(1, 2), (i, i)])
      @test rewards ≈ [0.5, 1.5] atol=1.e-6
      @test regret ≈ -0.5 atol=1.e-6
    end

    @testset "Check feasibility with 3 nodes on each side" begin
      n = 3
      μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      @test is_feasible(instance, [(1, 2), (2, 3), (3, 1)])
      @test ! is_feasible(instance, [(2, 3), (3, 2), (1, 2)])
      @test ! is_feasible(instance, [(2, 3), (3, 2), (2, 1), (1, 2)])
    end

    @testset "Check partial acceptability with 3 nodes on each side" begin
      n = 3
      μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)
      instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingNoSolver())

      @test is_partially_acceptable(instance, [(1, 2)])
      @test is_partially_acceptable(instance, [(1, 2), (2, 3)])
      @test is_partially_acceptable(instance, [(1, 2), (2, 3), (3, 1)])
      @test ! is_partially_acceptable(instance, [(2, 3), (3, 2), (1, 2)])
      @test ! is_partially_acceptable(instance, [(2, 3), (3, 2), (2, 1), (1, 2)])
    end

    if ! is_travis
      @testset "LP solver" begin
        @testset "Constructor" for i in [2, 5, 10]
          n = i
          μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
          Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
          reward = MvNormal(μ, Σ)
          instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))

          @test instance.solver != nothing
          @test instance.solver.model != nothing
          @test size(instance.solver.x, 1) == n * n
        end

        @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
          n = i
          μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
          Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
          reward = MvNormal(μ, Σ)
          instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))

          Random.seed!(i)
          drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
          solution = solve_linear(instance, drawn)
          @test is_feasible(instance, solution)
        end
      end
    end

    @testset "Munkres solver" begin
      @testset "Constructor" for i in [2, 5, 10]
        n = i
        μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
        Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
        reward = MvNormal(μ, Σ)
        instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

        @test instance.solver != nothing
      end

      @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
        n = i
        μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
        Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
        reward = MvNormal(μ, Σ)
        instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

        Random.seed!(i)
        drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
        solution = solve_linear(instance, drawn)
        @test is_feasible(instance, solution)
      end
    end

    @testset "Hungarian solver" begin
      @testset "Constructor" for i in [2, 5, 10]
        n = i
        μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
        Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
        reward = MvNormal(μ, Σ)
        instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

        @test instance.solver != nothing
      end

      @testset "Solve with $i nodes on each side" for i in [2, 5, 10]
        n = i
        μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
        Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
        reward = MvNormal(μ, Σ)
        instance = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

        Random.seed!(i)
        drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
        solution = solve_linear(instance, drawn)
        @test is_feasible(instance, solution)
      end
    end

    @testset "Solver equivalence (size: $i nodes on each side)" for i in [2, 5, 10]
      n = i
      μ = vec(Float64[.5 + ((i == j) ? 1. : 0.) for i in 1:n, j in 1:n])
      Σ = vec(Float64[((abs(i - j) <= 1) ? sign(i - j) : 0.) for i in 1:n, j in 1:n])
      reward = MvNormal(μ, Σ)

      instance_lp = ! is_travis && CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Gurobi.Optimizer))
      instance_munkres = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())
      instance_hungarian = CorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingHungarianSolver())

      Random.seed!(i)
      drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)

      if ! is_travis
        solution_lp = solve_linear(instance_lp, drawn)
        @test is_feasible(instance_lp, solution_lp)
      end

      solution_munkres = solve_linear(instance_munkres, drawn)
      @test is_feasible(instance_munkres, solution_munkres)

      solution_hungarian = solve_linear(instance_hungarian, drawn)
      @test is_feasible(instance_hungarian, solution_hungarian)

      # All solutions must have the same length, as these are perfect matchings.
      @test length(solution_hungarian) == length(solution_munkres)

      cost_munkres = sum(drawn[o] for o in solution_munkres)
      cost_hungarian = sum(drawn[o] for o in solution_hungarian)

      @test cost_hungarian ≈ cost_munkres

      if ! is_travis
        @test length(solution_lp) == length(solution_hungarian)
        cost_lp = sum(drawn[o] for o in solution_lp)
        @test cost_lp ≈ cost_hungarian
      end
    end
  end
end
