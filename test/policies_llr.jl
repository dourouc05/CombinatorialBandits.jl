@testset "LLR" begin
  @testset "Algorithm" begin
    n = 3
    ε = .1
    reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Cbc.Optimizer))

    # TODO: Cbc seems to output garbage for n_rounds - 1 iterations (probably because it does not support hot starts).
    Random.seed!(1)
    n_rounds = 20
    s = simulate(instance, LLR(), n_rounds)
    @test s.round == n_rounds
    @test sum(values(s.arm_counts)) == n_rounds * n
    @test s.regret ≈ 3.7
    @test s.reward ≈ 35.0
    @test s.regret + s.reward ≈ 38.7

    for arm in keys(s.arm_counts)
      @test s.arm_average_reward[arm] ≈ s.arm_reward[arm] / s.arm_counts[arm]
    end
  end

  @testset "Trace" begin
    n = 3
    ε = .1
    reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(Cbc.Optimizer))

    n_rounds = 20
    test_policy_trace(instance, LLR(), n_rounds, n_rounds * n, 3.7, 35.0, 38.7)
  end
end
