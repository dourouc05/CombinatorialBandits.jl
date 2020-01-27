@testset "Shortest paths" begin
  @testset "Constructor with $i nodes" for i in [2, 5, 10]
    # Edge weights: high gain likelihood for the path 1->2->3...n, low for the other edges.
    n = i
    ε = 1 / (n + 1)
    graph = complete_digraph(n)
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
      for i in 1:n, j in 1:n if i != j)
    instance = ElementaryPath(graph, reward, 1, n, ElementaryPathNoSolver())

    @test instance.n_arms == ne(graph)
    @test instance.reward == reward

    # Error: source makes no sense.
    @test_throws ErrorException ElementaryPath(graph, reward, - 1, n, ElementaryPathNoSolver())
    @test_throws ErrorException ElementaryPath(graph, reward, n + 1, n, ElementaryPathNoSolver())

    # Error: destination makes no sense.
    @test_throws ErrorException ElementaryPath(graph, reward, 1, - 1, ElementaryPathNoSolver())
    @test_throws ErrorException ElementaryPath(graph, reward, 1, n + 1, ElementaryPathNoSolver())
  end

  @testset "State with $i nodes" for i in [2, 5, 10]
    n = i
    ε = 1 / (n + 1)
    graph = complete_digraph(n)
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
      for i in 1:n, j in 1:n if i != j)
    instance = ElementaryPath(graph, reward, 1, n, ElementaryPathNoSolver())

    state = initial_state(instance)

    @test state.round == 0
    @test state.regret == 0.0
    @test state.reward == 0.0
    @test length(state.arm_counts) == ne(graph)
    @test length(state.arm_reward) == ne(graph)
    @test length(state.arm_average_reward) == ne(graph)

    for i in 1:n
      for j in 1:n
        if i == j
          continue
        end

        @test state.arm_counts[(i, j)] == 0
        @test state.arm_reward[(i, j)] == 0.0
        @test state.arm_average_reward[(i, j)] == 0.0
      end
    end
  end

  @testset "Trace with $i nodes" for i in [2, 5, 10]
    n = i
    ε = 1 / (n + 1)
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
      for i in 1:n, j in 1:n if i != j)
    instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathNoSolver())

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

  @testset "Pull with $i nodes" for i in [2, 5, 10]
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i == 1 && j == i) ? 1 : 1 / i)
      for i in 1:i, j in 1:i if i != j)
    instance = ElementaryPath(complete_digraph(i), reward, 1, i, ElementaryPathLightGraphsDijkstraSolver())

    Random.seed!(2)
    reward, regret = pull(instance, [(1, i)])
    @test reward ≈ [1.0]
    @test regret ≈ 0.0
  end

  @testset "Check feasibility with 3 nodes" begin
    n = 3
    ε = 1 / (n + 1)
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
      for i in 1:n, j in 1:n if i != j)
    instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathNoSolver())

    @test is_feasible(instance, [(1, 3)])
    @test is_feasible(instance, [(1, 2), (2, 3)])
    @test ! is_feasible(instance, Tuple{Int, Int}[])
    @test ! is_feasible(instance, [(1, 2), (2, 3), (3, 1)])
    @test ! is_feasible(instance, [(2, 3), (3, 2), (1, 2)])
    @test ! is_feasible(instance, [(2, 3), (3, 2), (2, 1), (1, 2)])
  end

  @testset "Check partial acceptability with 3 nodes" begin
    n = 3
    ε = 1 / (n + 1)
    reward = Dict{Tuple{Int64, Int64}, Distribution}(
      (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
      for i in 1:n, j in 1:n if i != j)
    instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathNoSolver())

    @test is_partially_acceptable(instance, [(1, 3)])
    @test is_partially_acceptable(instance, [(1, 2)])
    @test is_partially_acceptable(instance, [(1, 2), (2, 3)])
    @test is_partially_acceptable(instance, Tuple{Int, Int}[])
    @test ! is_partially_acceptable(instance, [(1, 2), (2, 3), (3, 1)])
    @test ! is_partially_acceptable(instance, [(2, 3), (3, 2), (1, 2)])
    @test ! is_partially_acceptable(instance, [(2, 3), (3, 2), (2, 1), (1, 2)])
  end

  @testset "LightGraphs.jl Dijkstra solver" begin
    @testset "Constructor with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathLightGraphsDijkstraSolver())

      @test instance.solver != nothing
      @test instance.solver.graph != nothing
      @test size(instance.solver.weights_matrix, 1) == n
      @test size(instance.solver.weights_matrix, 2) == n
      @test instance.solver.source == 1
      @test instance.solver.destination == n
    end

    @testset "Solve with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathLightGraphsDijkstraSolver())

      Random.seed!(i)
      drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n)
      solution = solve_linear(instance, drawn)
      @test is_feasible(instance, solution)
    end
  end

  @testset "LP solver" begin
    @testset "Constructor with $i nodes" for i in [5]#[2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathLPSolver(Gurobi.Optimizer))

      @test instance.solver != nothing
      @test instance.solver.graph != nothing
      @test instance.solver.source == 1
      @test instance.solver.destination == n
    end

    @testset "Solve with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathLPSolver(Gurobi.Optimizer))

      Random.seed!(i)
      drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n if i != j)
      solution = solve_linear(instance, drawn)
      @test is_feasible(instance, solution)
    end
  end

  @testset "Algos solver" begin
    @testset "Constructor with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathAlgosSolver())

      @test instance.solver != nothing
      @test instance.solver.graph != nothing
      @test instance.solver.source == 1
      @test instance.solver.destination == n
    end

    @testset "Solve with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i + 1 == j) ? (1 - ε) : (ε))
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathAlgosSolver())

      Random.seed!(i)
      drawn = Dict((i, j) => rand() for i in 1:n, j in 1:n if i != j)
      solution = solve_linear(instance, drawn)
      @test is_feasible(instance, solution)
    end

    @testset "Solve with $i nodes and a budget" for i in [3]# [2, 5, 10]
      # Test case: all edges have a 100% probability of reward and a zero weight,
      # except the direct edge from the source to the destination (no reward,
      # some weight). The algorithm is looking for a path with a nonzero minimum
      # weight, i.e. there is only one possible solution (1 -> n) with zero reward.
      n = i
      ε = 1 / (n + 1)
      reward = Dict{Tuple{Int64, Int64}, Distribution}(
        (i, j) => Bernoulli((i == 1 && j == n) ? 0 : 1)
        for i in 1:n, j in 1:n if i != j)
      instance = ElementaryPath(complete_digraph(n), reward, 1, n, ElementaryPathAlgosSolver())

      # There is no real randomness in this test case, hence no need to set the seed.
      rewards = Dict((i, j) => Float64(rand(reward[i, j])) for i in 1:n, j in 1:n if i != j)
      weights = Dict((i, j) => (i == 1 && j == n) ? 2 : 0 for i in 1:n, j in 1:n if i != j)
      solution = solve_budgeted_linear(instance, rewards, weights, 2)
      @test solution == [(1, n)]
    end
  end

  @testset "Solver equivalence" begin
    function compare_lp_algos(graph, s, d, rewards, weights, ε, budgets)
      ε = 0.003
      rewards_discrete = Dict(k => round(Int, v / ε, RoundUp) for (k, v) in rewards)
      distr = Dict{Tuple{Int, Int}, Distribution}((src(e), dst(e)) => Bernoulli(0.0) for e in edges(graph))

      @testset "Budget $budget" for budget in budgets
        i_lp = ElementaryPath(graph, distr, s, d, ElementaryPathLPSolver(Gurobi.Optimizer))
        i_al = ElementaryPath(graph, distr, s, d, ElementaryPathAlgosSolver())
        s_lp = solve_budgeted_linear(i_lp.solver, weights, rewards_discrete, budget)
        s_al = solve_budgeted_linear(i_al.solver, weights, rewards_discrete, budget)

        if length(s_lp) == 0
          @test length(s_lp) == length(s_al)
          continue
        end
        @test length(s_lp) >= 1
        @test length(s_al) >= 1

        @test sum(rewards_discrete[arm] for arm in s_lp) >= budget
        @test sum(rewards_discrete[arm] for arm in s_al) >= budget

        o_lp = sum(rewards[arm] for arm in s_lp) + sqrt(sum(weights[arm] for arm in s_lp))
        o_al = sum(rewards[arm] for arm in s_al) + sqrt(sum(weights[arm] for arm in s_al))
        @test o_al ≈ o_lp atol=ε

        # Compare directly against Bellman-Ford.
        if budget == 0
          weights_edge = Dict(Edge(k...) => v for (k, v) in weights)
          isp = CombinatorialBandits.ElementaryPathInstance(graph, weights_edge, s, d)
          s_bf = CombinatorialBandits.lp_dp(isp).path
          o_bf = sum(rewards[src(arm), dst(arm)] for arm in s_bf) + sqrt(sum(weights[src(arm), dst(arm)] for arm in s_bf))

          @test o_al ≈ o_bf atol=ε
        end
      end

      @testset "Overall solution" begin
        i_lp = ElementaryPath(graph, distr, s, d, ElementaryPathLPSolver(Gurobi.Optimizer))
        i_al = ElementaryPath(graph, distr, s, d, ElementaryPathAlgosSolver())
        s_lp = CombinatorialBandits.optimise_linear_sqrtlinear(i_lp, ESCB2Exact(), rewards, weights)
        s_al = CombinatorialBandits.optimise_linear_sqrtlinear(i_al, ESCB2Budgeted(ε, true), rewards, weights)
        o_lp = sum(rewards[arm] for arm in s_lp) + sqrt(sum(weights[arm] for arm in s_lp))
        o_al = sum(rewards[arm] for arm in s_al) + sqrt(sum(weights[arm] for arm in s_al))

        @test o_lp ≈ o_al atol=ε
      end
    end

    function get_graph(n)
      graph = DiGraph(n)
      for i in 1:(n - 1)
        for j in (i + 1):n
          add_edge!(graph, i, j)
        end
      end
      return graph, 1, n
    end

    @testset "First case" begin
      graph, s, d = get_graph(5)
      ε = 0.003
      rewards = Dict((1, 2) => 0.0,(1, 3) => 0.0,(1, 4) => 1.0,(1, 5) => 1.0,(2, 3) => 0.3333333333333333,(2, 4) => 0.0,(2, 5) => 0.0,(3, 4) => 0.3333333333333333,(3, 5) => 0.0,(4, 5) => 0.0)
      weights = Dict((1, 3) => 15.681708132568822,(1, 2) => 3.1363416265137642,(1, 4) => 15.681708132568822,(1, 5) => 15.681708132568822,(2, 3) => 5.227236044189607,(2, 4) => 15.681708132568822,(2, 5) => 15.681708132568822,(3, 4) => 5.227236044189607,(3, 5) => 15.681708132568822,(4, 5) => 3.1363416265137642)

      compare_lp_algos(graph, s, d, rewards, weights, ε, [0, 1, 112, 113, 224, 225, 334, 335])
       # No more solution after 334
    end

    @testset "Second case" begin
      graph, s, d = get_graph(5)
      ε = 0.003
      rewards = Dict((1, 2) => 0.18126888217522658,(1, 3) => 0.1527777777777778,(1, 4) => 0.23404255319148937,(1, 5) => 0.8253968253968254,(2, 3) => 0.13100436681222707,(2, 4) => 0.12307692307692308,(2, 5) => 0.10810810810810811,(3, 4) => 0.14847161572052403,(3, 5) => 0.16666666666666666,(4, 5) => 0.16129032258064516)
      weights = Dict((1, 2) => 0.12134857271805984,(1, 3) => 0.557866355134414,(1, 4) => 0.8546037780782512,(1, 5) => 0.3187807743625223,(2, 3) => 0.17539902868854937,(2, 4) => 0.6179442703027355,(2, 5) => 1.0855777721534543,(3, 4) => 0.17539902868854937,(3, 5) => 0.557866355134414,(4, 5) => 0.11778996354744224)

      compare_lp_algos(graph, s, d, rewards, weights, ε, [0, 1, 37, 38, 42, 43, 44, 45, 50, 51, 52, 54, 55, 56, 61, 62, 79, 80, 82, 83, 87, 88, 89, 91, 92, 93, 94, 95, 98, 99, 276, 277, 552, 553, 828, 829])
    end

    @testset "Third case" begin
      graph, s, d = get_graph(10)
      ε = 1/63
      rewards = Dict((1, 2) => 0.08148148148148149,(1, 3) => 0.0958904109589041,(1, 4) => 0.09090909090909091,(1, 5) => 0.05,(1, 6) => 0.125,(1, 7) => 0.0,(1, 8) => 0.09090909090909091,(1, 9) => 0.2,(1, 10) => 0.0, (2, 3) => 0.07746478873239436,(2, 4) => 0.02631578947368421,(2, 5) => 0.11538461538461539,(2, 6) => 0.1111111111111111,(2, 7) => 0.2,(2, 8) => 0.0,(2, 9) => 0.2727272727272727,(2, 10) => 0.0,(3, 4) => 0.14953271028037382,(3, 5) => 0.0625,(3, 6) => 0.2,(3, 7) => 0.0,(3, 8) => 0.07692307692307693,(3, 9) => 0.25,(3, 10) => 0.09090909090909091,(4, 5) => 0.11764705882352941,(4, 6) => 0.0625,(4, 7) => 0.0,(4, 8) => 0.0625,(4, 9) => 0.07142857142857142,(4, 10) => 0.0,(5, 6) => 0.06666666666666667,(5, 7) => 0.09090909090909091,(5, 8) => 0.0, (5, 9) => 0.05555555555555555,(5, 10) => 0.125,(6, 7) => 0.14772727272727273,(6, 8) => 0.06060606060606061,(6, 9) => 0.041666666666666664,(6, 10) => 0.047619047619047616,(7, 8) => 0.12149532710280374,(7, 9) => 0.047619047619047616,(7, 10) => 0.08823529411764706,(8, 9) => 0.06716417910447761,(8, 10) => 0.11538461538461539,(9, 10) => 0.0830188679245283)
      weights = Dict((1, 2) => 0.6156966918752366,(1, 3) => 2.277234339812519,(1, 4) => 5.037518388070118,(1, 5) => 8.311905340315693,(1, 6) => 10.389881675394617,(1, 7) => 12.78754667740876,(1, 8) => 15.112555164210352,(1, 9) => 16.623810680631387,(1, 10) => 12.78754667740876,(2, 3) => 1.1706908930022104,(2, 4) => 4.3746870212187865,(2, 5) => 6.39377333870438,(2, 6) => 9.235450378128549,(2, 7) => 11.082540453754259,(2, 8) => 15.112555164210352,(2, 9) => 15.112555164210352,(2, 10) => 18.470900756257098,(3, 4) => 1.5536271664141483,(3, 5) => 5.194940837697309,(3, 6) => 6.649524272252555,(3, 7) => 11.082540453754259,(3, 8) => 12.78754667740876,(3, 9) => 13.853175567192823,(3, 10) => 15.112555164210352,(4, 5) => 1.9557424330154574,(4, 6) => 5.194940837697309,(4, 7) => 8.749374042437573,(4, 8) => 10.389881675394617,(4, 9) => 11.874150486165277,(4, 10) => 13.853175567192823,(5, 6) => 2.216508090750852,(5, 7) => 5.037518388070118,(5, 8) => 7.916100324110185,(5, 9) => 9.235450378128549,(5, 10) => 10.389881675394617,(6, 7) => 1.889069395526294,(6, 8) => 5.037518388070118,(6, 9) => 6.926587783596411,(6, 10) => 7.916100324110185,(7, 8) => 1.5536271664141483,(7, 9) => 3.9580501620550925,(7, 10) => 4.889356082538644,(8, 9) => 1.2405828866142827,(8, 10) => 2.1312577795681267,(9, 10) => 0.6273136105898637)

      compare_lp_algos(graph, s, d, rewards, weights, ε, collect(0:20))
    end
  end
end
