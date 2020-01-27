@testset "Minimum spanning trees in a complete graph" begin
  @testset "Constructor with $i nodes" for i in [2, 5, 10]
    # Edge weights: high gain likelihood for the edges 1->2->3...n, low for the other edges.
    n = i
    ε = 1 / n
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeNoSolver())

    @test instance.n_arms == n * (n - 1) / 2
    @test instance.reward == reward
  end

  @testset "State with $i nodes" for i in [2, 5, 10]
    n = i
    ε = 1 / n
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeNoSolver())

    state = initial_state(instance)

    @test state.round == 0
    @test state.regret == 0.0
    @test state.reward == 0.0
    @test length(state.arm_counts) == ne(graph)
    @test length(state.arm_reward) == ne(graph)
    @test length(state.arm_average_reward) == ne(graph)

    for e in keys(reward)
      @test state.arm_counts[e...] == 0
      @test state.arm_reward[e...] == 0.0
      @test state.arm_average_reward[e...] == 0.0
    end
  end

  @testset "Trace with $i nodes" for i in [2, 5, 10]
    n = i
    ε = 1 / n
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeNoSolver())

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
    n = i
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) == 1 && dst(e) == n) ? 1 : 0)
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeLightGraphsPrimSolver())

    Random.seed!(1)
    @test pull(instance, [(1, i)]) == ([1.0], 0.0)
  end

  @testset "Check feasibility with 3 nodes" begin
    n = 3
    ε = 1 / n
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeNoSolver())

    @test ! is_feasible(instance, [(1, 3)])
    @test is_feasible(instance, [(1, 2), (2, 3)])
    @test is_feasible(instance, [(1, 3), (1, 3)])
    @test is_feasible(instance, [(2, 1), (2, 3)])
    @test ! is_feasible(instance, Tuple{Int, Int}[])
    @test ! is_feasible(instance, [(1, 2), (2, 3), (3, 1)])
  end

  @testset "Check partial acceptability with 3 nodes" begin
    n = 3
    ε = 1 / n
    graph = complete_graph(n)
    reward = Dict{Tuple{Int, Int}, Distribution}(
      (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
      for e in edges(graph))
    instance = SpanningTree(graph, reward, SpanningTreeNoSolver())

    @test is_partially_acceptable(instance, [(1, 3)])
    @test is_partially_acceptable(instance, [(1, 2), (2, 3)])
    @test is_partially_acceptable(instance, [(1, 3), (1, 2)])
    @test is_partially_acceptable(instance, [(2, 1), (2, 3)])
    @test is_partially_acceptable(instance, Tuple{Int, Int}[])
    @test ! is_partially_acceptable(instance, [(1, 3), (1, 3)])
    @test ! is_partially_acceptable(instance, [(1, 2), (2, 3), (3, 1)])
    @test ! is_partially_acceptable(instance, [(1, 2), (2, 1)])
  end

  @testset "LightGraphs.jl Prim solver" begin
    @testset "Constructor" for i in [2, 3]#, 5, 10]
      n = i
      ε = 1 / n
      graph = complete_graph(n)
      reward = Dict{Tuple{Int, Int}, Distribution}(
        (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
        for e in edges(graph))
      instance = SpanningTree(graph, reward, SpanningTreeLightGraphsPrimSolver())

      @test instance.solver != nothing
    end

    @testset "Solve with $i nodes" for i in [2, 5, 10]
      n = i
      ε = 1 / (n + 1)
      graph = complete_graph(n)
      reward = Dict{Tuple{Int, Int}, Distribution}(
        (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
        for e in edges(graph))
      instance = SpanningTree(graph, reward, SpanningTreeLightGraphsPrimSolver())

      Random.seed!(i)
      drawn = Dict(k => rand() for (k, _) in reward)
      solution = solve_linear(instance, drawn)
      @test is_feasible(instance, solution)
    end
  end

  if ! is_travis
    @testset "LP solver" begin
      @testset "Constructor" for i in [2, 5, 10]
        n = i
        ε = 1 / n
        graph = complete_graph(n)
        reward = Dict{Tuple{Int, Int}, Distribution}(
          (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
          for e in edges(graph))
        instance = SpanningTree(graph, reward, SpanningTreeLPSolver(Gurobi.Optimizer))

        @test instance.solver != nothing
      end

      @testset "Solve with $i nodes" for i in [2, 5, 10]
        n = i
        ε = 1 / (n + 1)
        graph = complete_graph(n)
        reward = Dict{Tuple{Int, Int}, Distribution}(
          (src(e), dst(e)) => Bernoulli((src(e) + 1 == dst(e)) ? (1 - ε) : (ε))
          for e in edges(graph))
        instance = SpanningTree(graph, reward, SpanningTreeLPSolver(Gurobi.Optimizer))

        Random.seed!(i)
        drawn = Dict(k => rand() for (k, _) in reward)
        solution = solve_linear(instance, drawn)
        @test is_feasible(instance, solution)
      end
    end
  end
end
