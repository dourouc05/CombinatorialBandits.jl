@testset "Thompson sampling" begin
  @testset "Algorithm" begin
    n = 3
    ε = .1
    reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

    Random.seed!(1)
    n_rounds = 20
    s = simulate(instance, ThompsonSampling(), n_rounds)
    @test s.round == n_rounds
    @test sum(values(s.arm_counts)) == n_rounds * n
    @test s.regret ≈ 4.2
    @test s.reward ≈ 30.0
    @test s.regret + s.reward ≈ 34.2

    for arm in keys(s.arm_counts)
      @test s.arm_average_reward[arm] ≈ s.arm_reward[arm] / s.arm_counts[arm]
    end
  end

  @testset "Trace" begin
    n = 3
    ε = .1
    reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

    n_rounds = 20
    test_policy_trace(instance, ThompsonSampling(), n_rounds, n_rounds * n, 4.2, 30.0, 34.2)
  end
end
