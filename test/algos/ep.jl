using Test

@testset "Elementary paths" begin
  @testset "Interface" begin
    @testset "Positive-reward cycle" begin
      g = path_digraph(3)
      add_edge!(g, 2, 1)
      costs = Dict(Edge(1, 2) => 1.0, Edge(2, 3) => 3.0, Edge(2, 1) => 25.0)

      i = ElementaryPathInstance(g, costs, 1, 3)
      @test_logs (:warn, "The graph contains a positive-cost cycle around edge 2 -> 1.") lp_dp(i)
    end
  end

  @testset "Basic" begin
    g = path_digraph(3)
    costs = Dict(Edge(1, 2) => 1.0, Edge(2, 3) => 1.0)
    i = ElementaryPathInstance(g, costs, 1, 3)
    d = lp_dp(i)

    @test d.instance == i
    @test d.path == [Edge(1, 2), Edge(2, 3)]

    add_edge!(g, 1, 3)
    costs[Edge(1, 3)] = 3.0
    i = ElementaryPathInstance(g, costs, 1, 3)
    d = lp_dp(i)

    @test d.instance == i
    @test d.path == [Edge(1, 3)]
  end
end

@testset "Budgeted elementary paths" begin
  @testset "Interface" begin
    @testset "Constructor" begin
      g = path_digraph(3)
      rewards = Dict(Edge(1, 2) => 1.0, Edge(2, 3) => 1.0)
      weights = Dict(Edge(1, 2) => 1, Edge(2, 3) => 1)
      @test_throws ErrorException BudgetedElementaryPathInstance(g, rewards, weights, 1, 1)
      @test_throws ErrorException BudgetedElementaryPathInstance(g, rewards, weights, 1, 2, budget=-1)

      rewards[Edge(4, 1)] = 1.0
      @test_throws ErrorException BudgetedElementaryPathInstance(g, rewards, weights, 1, 1)
      weights[Edge(4, 1)] = 1
      weights[Edge(4, 2)] = 1
      @test_throws ErrorException BudgetedElementaryPathInstance(g, rewards, weights, 1, 1)
      weights[Edge(4, 2)] = -1
      @test_throws ErrorException BudgetedElementaryPathInstance(g, rewards, weights, 1, 1)
    end
  end

  @testset "Basic" begin
    # Create a directed path graph.
    g = path_digraph(3)
    rewards = Dict(Edge(1, 2) => 1.0, Edge(2, 3) => 1.0)
    weights = Dict(Edge(1, 2) => 1, Edge(2, 3) => 1)
    i = BudgetedElementaryPathInstance(g, rewards, weights, 1, 3, budget=4, max_weight=1)
    d = budgeted_lp_dp(i)

    @test d.instance == i
    @test d.path == [] # No path with a total weight of at least 4.

    for β in [0, 1, 2]
      @test d.solutions[1, β] == []
      @test d.solutions[2, β] == ((β == 2) ? [] : [Edge(1, 2)])
      @test d.solutions[3, β] == [Edge(1, 2), Edge(2, 3)]
    end
    for β in [3, 4]
      for v in [1, 2, 3]
        @test d.solutions[v, β] == []
      end
    end

    add_edge!(g, 1, 3)
    rewards[Edge(1, 3)] = 1.0
    weights[Edge(1, 3)] = 4
    i = BudgetedElementaryPathInstance(g, rewards, weights, 1, 3, budget=4, max_weight=1)
    d = budgeted_lp_dp(i)

    @test d.instance == i
    @test d.path == [Edge(1, 3)]

    warn_msg = "The asked maximum budget 5 is higher than the instance budget 4. Therefore, some values have not been computed and are unavailable."
    @test_logs (:warn, warn_msg) CombinatorialBandits.paths_all_budgets_as_tuples(d, 5)
    sol = Dict(0 => [(1, 2), (2, 3)], 1 => [(1, 2), (2, 3)], 2 => [(1, 2), (2, 3)], 3 => [(1, 3)], 4 => [(1, 3)])
    @test CombinatorialBandits.paths_all_budgets_as_tuples(d, 4) == sol

    @test_logs (:warn, warn_msg) CombinatorialBandits.paths_all_budgets(d, 5)
    sol = Dict(0 => [Edge(1, 2), Edge(2, 3)], 1 => [Edge(1, 2), Edge(2, 3)], 2 => [Edge(1, 2), Edge(2, 3)], 3 => [Edge(1, 3)], 4 => [Edge(1, 3)])
    @test CombinatorialBandits.paths_all_budgets(d, 4) == sol
  end

  @testset "Conformity" begin
    # More advanced tests to ensure the algorithm works as expected.
    g = complete_digraph(3)
    rewards = Dict(Edge(1, 2) => 1.0, Edge(3, 1) => -1.0, Edge(3, 2) => -1.0, Edge(2, 3) => 1.0, Edge(2, 1) => -1.0, Edge(1, 3) => 0.0)
    weights = Dict(Edge(1, 2) => 0, Edge(3, 1) => 0, Edge(3, 2) => 0, Edge(2, 3) => 0, Edge(2, 1) => 0, Edge(1, 3) => 2)
    i = BudgetedElementaryPathInstance(g, rewards, weights, 1, 3, budget=2, max_weight=1)
    warn_msg = "The graph contains a positive-cost cycle around edge 3 -> 1."
    d = @test_logs (:warn, warn_msg) budgeted_lp_dp(i)

    @test d.path == [Edge(1, 3)]

    @test d.solutions[1, 0] == []
    @test d.solutions[2, 0] == [Edge(1, 2)]
    @test d.solutions[3, 0] == [Edge(1, 2), Edge(2, 3)]
    @test d.states[1, 0] == 0.0
    @test d.states[2, 0] == 1.0
    @test d.states[3, 0] == 2.0

    @test d.solutions[1, 1] == []
    @test d.solutions[2, 1] == [Edge(1, 3), Edge(3, 2)]
    @test d.solutions[3, 1] == [Edge(1, 3)]
    @test d.states[1, 1] == 0.0
    @test d.states[2, 1] == -1.0
    @test d.states[3, 1] == 0.0

    @test d.solutions[1, 2] == []
    @test d.solutions[2, 2] == [Edge(1, 3), Edge(3, 2)]
    @test d.solutions[3, 2] == [Edge(1, 3)]
    @test d.states[1, 2] == 0.0
    @test d.states[2, 2] == -1.0
    @test d.states[3, 2] == 0.0
  end
end
