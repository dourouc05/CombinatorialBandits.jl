@testset "ESCB2" begin
  n = 3
  ε = .1
  reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]

  if ! is_travis
    @testset "Exact" begin
      instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(CPLEX.Optimizer))

      @testset "Algorithm" begin
        Random.seed!(1)
        n_rounds = 20
        s = simulate(instance, ESCB2(ESCB2Exact()), n_rounds)
        @test s.round == n_rounds
        @test sum(values(s.arm_counts)) <= n_rounds * n
        @test s.regret ≈ 3.2
        @test s.reward ≈ 31.0
        @test s.regret + s.reward ≈ 34.2

        for arm in keys(s.arm_counts)
          @test s.arm_average_reward[arm] ≈ s.arm_reward[arm] / s.arm_counts[arm]
        end
      end

      @testset "Trace" begin
        n = 3
        ε = .1
        reward = Distribution[Bernoulli(.5 + ((i == j) ? ε : 0.)) for i in 1:n, j in 1:n]
        instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingLPSolver(CPLEX.Optimizer))

        n_rounds = 20
        test_policy_trace(instance, ESCB2(ESCB2Exact()), n_rounds, n_rounds * n, 3.2, 31.0, 34.2)
      end
    end
  end

  @testset "Greedy" begin
    instance = UncorrelatedPerfectBipartiteMatching(reward, PerfectBipartiteMatchingMunkresSolver())

    @testset "Algorithm" begin
      Random.seed!(1)
      n_rounds = 20
      s = simulate(instance, ESCB2(ESCB2Greedy()), n_rounds)
      @test s.round == n_rounds
      @test sum(values(s.arm_counts)) == n_rounds * n
      @test s.regret ≈ 3.8
      @test s.reward ≈ 30.0
      @test s.regret + s.reward ≈ 33.8

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
      test_policy_trace(instance, ESCB2(ESCB2Greedy()), n_rounds, n_rounds * n, 3.8, 30.0, 33.8)
    end
  end

  @testset "Budgeted (all at once)" begin
    # Not available for many instances, for now…
    m = 2
    r = Distribution[Bernoulli(.5 + ((i % 2 == 0) ? ε : 0.)) for i in 1:n]
    i = MSet(r, m, MSetAlgosSolver())

    @testset "Algorithm" begin
      Random.seed!(1)
      n_rounds = 20
      s = simulate(i, ESCB2(ESCB2Budgeted(.1, true)), n_rounds)
      @test s.round == n_rounds
      @test sum(values(s.arm_counts)) == n_rounds * m
      @test s.regret ≈ 0.9
      @test s.reward ≈ 25.0
      @test s.regret + s.reward ≈ 25.9

      for arm in keys(s.arm_counts)
        @test s.arm_average_reward[arm] ≈ s.arm_reward[arm] / s.arm_counts[arm]
      end
    end

    @testset "Trace" begin
      n_rounds = 20
      test_policy_trace(i, ESCB2(ESCB2Budgeted(.1, true)), n_rounds, n_rounds * m, 0.9, 25.0, 25.9)
    end
  end

  @testset "Budgeted (one by one)" begin
    # Not available for many instances, for now…
    m = 2
    r = Distribution[Bernoulli(.5 + ((i % 2 == 0) ? ε : 0.)) for i in 1:n]
    i = MSet(r, m, MSetAlgosSolver())

    @testset "Algorithm" begin
      Random.seed!(1)
      n_rounds = 20
      s = simulate(i, ESCB2(ESCB2Budgeted(.1, false)), n_rounds)
      @test s.round == n_rounds
      @test sum(values(s.arm_counts)) <= n_rounds * m
      @test s.regret ≈ 0.9
      @test s.reward ≈ 25.0
      @test s.regret + s.reward ≈ 25.9

      for arm in keys(s.arm_counts)
        @test s.arm_average_reward[arm] ≈ s.arm_reward[arm] / s.arm_counts[arm]
      end
    end

    @testset "Trace" begin
      n_rounds = 20
      test_policy_trace(i, ESCB2(ESCB2Budgeted(.1, false)), n_rounds, n_rounds * m, 0.9, 25.0, 25.9)
    end
  end

  @testset "optimise_linear_sqrtlinear" begin
    # Test data coming from 1000 rounds, with ε <= 1/sqrt(1000) ≈ 0.03.

    m = 2
    distr = Distribution[Bernoulli(.5) for i in 1:5]
    i = MSet(distr, m, MSetAlgosSolver())
    ilp = ! is_travis && MSet(distr, m, MSetLPSolver(Gurobi.Optimizer))
    w = Dict(1 => 0.16666666666666666,2 => 0.8041237113402062,3 => 0.16666666666666666,4 => 0.7921348314606742,5 => 0.16666666666666666)
    s2 = Dict(1 => 1.0818164611145353,2 => 0.10037472319619399,3 => 1.0818164611145353,4 => 0.10939717022506537,5 => 1.0818164611145353)
    exact_obj = 2.058076533676451

    algo = ESCB2Budgeted(.01, true)
    sol, rd = CombinatorialBandits.optimise_linear_sqrtlinear(i, algo, w, s2, with_trace=true)
    @test CombinatorialBandits.escb2_index(w, s2, sol) ≈ exact_obj
    @test rd.best_objective ≈ exact_obj

    if ! is_travis
      algo = ESCB2Exact()
      sol, rd = CombinatorialBandits.optimise_linear_sqrtlinear(ilp, algo, w, s2, with_trace=true)
      @test CombinatorialBandits.escb2_index(w, s2, sol) ≈ exact_obj
      @test rd.best_objective ≈ exact_obj
    end

    m = 25
    distr = Distribution[Bernoulli(.5) for i in 1:50]
    i = MSet(distr, m, MSetAlgosSolver())
    ilp = ! is_travis && MSet(distr, m, MSetLPSolver(Gurobi.Optimizer))
    w = Dict(1 => 0.24242424242424243,2 => 0.7777777777777778,3 => 0.35714285714285715,4 => 0.8048048048048048,5 => 0.21875,6 => 0.8308605341246291,7 => 0.13793103448275862,8 => 0.7685185185185185,9 => 0.21875,10 => 0.7689393939393939,11 => 0.1111111111111111,12 => 0.7692307692307693,13 => 0.24242424242424243,14 => 0.7877813504823151,15 => 0.04,16 => 0.7509433962264151,17 => 0.24242424242424243,18 => 0.827485380116959,19 => 0.07692307692307693,20 => 0.7813765182186235,21 => 0.2647058823529412,22 => 0.7592592592592593,23 => 0.13793103448275862,24 => 0.8245614035087719,25 => 0.1111111111111111,26 => 0.8054711246200608,27 => 0.16666666666666666,28 => 0.7746913580246914,29 => 0.1935483870967742,30 => 0.8033333333333333,31 => 0.2972972972972973,32 => 0.8230088495575221,33 => 0.21875,34 => 0.7781350482315113,35 => 0.24242424242424243,36 => 0.8011695906432749,37 => 0.13793103448275862,38 => 0.8176470588235294,39 => 0.1935483870967742,40 => 0.8159509202453987,41 => 0.07692307692307693,42 => 0.8147058823529412,43 => 0.2972972972972973,44 => 0.7466666666666667,45 => 0.21875,46 => 0.7704918032786885,47 => 0.16666666666666666,48 => 0.7647058823529411,49 => 0.21875,50 => 0.8224852071005917)
    s2 = Dict(1 => 5.434941671158229,2 => 0.6227537331535471,3 => 4.270311313052894,4 => 0.5385978232679326,5 => 5.604783598381924,6 => 0.5322049707662361,7 => 6.184588798214537,8 => 0.5535588739142641,9 => 5.604783598381924,10 => 0.6793677088947786,11 => 6.642706486971169,12 => 0.6898195198008522,13 => 5.434941671158229,14 => 0.5766979908302944,15 => 7.1741230059288625,16 => 0.6768040571631002,17 => 5.434941671158229,18 => 0.5244241963398292,19 => 6.898195198008522,20 => 0.7261258103166864,21 => 5.275090445535929,22 => 0.5535588739142641,23 => 6.184588798214537,24 => 0.5244241963398292,25 => 6.642706486971169,26 => 0.5451461250705822,27 => 5.978435838274052,28 => 0.5535588739142641,29 => 5.78558306929747,30 => 0.5978435838274052,31 => 4.847380409411394,32 => 0.5290651184313321,33 => 5.604783598381924,34 => 0.5766979908302944,35 => 5.434941671158229,36 => 0.5244241963398292,37 => 6.184588798214537,38 => 0.5275090445535928,39 => 5.78558306929747,40 => 0.5501628072031336,41 => 6.898195198008522,42 => 0.5275090445535928,43 => 4.847380409411394,44 => 0.5978435838274052,45 => 5.604783598381924,46 => 0.7350535866730392,47 => 5.978435838274052,48 => 0.5552726784774661,49 => 5.604783598381924,50 => 0.5306303998468094)
    exact_obj = 23.701163713122835

    ε = 0.014581
    algo = ESCB2Budgeted(ε, true)
    sol, rd = CombinatorialBandits.optimise_linear_sqrtlinear(i, algo, w, s2, with_trace=true)
    @test CombinatorialBandits.escb2_index(w, s2, sol) ≈ exact_obj atol=ε
    @test rd.best_objective ≈ exact_obj atol=ε

    if ! is_travis
      algo = ESCB2Exact()
      sol, rd = CombinatorialBandits.optimise_linear_sqrtlinear(ilp, algo, w, s2, with_trace=true)
      @test CombinatorialBandits.escb2_index(w, s2, sol) ≈ exact_obj
      @test rd.best_objective ≈ exact_obj
    end
  end
end
