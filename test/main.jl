@testset "Main" begin
  @testset "Simulate with trace" begin
    n = 3
    ε = .1
    reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

    Random.seed!(1)
    n_rounds = 2
    s, t = simulate(instance, ThompsonSampling(), n_rounds, with_trace=true)
    @test s.round == n_rounds

    @test length(t.states) == n_rounds
    @test length(t.arms) == n_rounds
    @test length(t.reward) == n_rounds

    @test t.states[n_rounds].round == s.round
    @test t.states[n_rounds].regret == s.regret
    @test t.states[n_rounds].reward == s.reward
    @test t.states[n_rounds].arm_counts == s.arm_counts
    @test t.states[n_rounds].arm_reward == s.arm_reward
    @test t.states[n_rounds].arm_average_reward == s.arm_average_reward
  end
end
