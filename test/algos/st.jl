using Test

@testset "Maximum spanning tree" begin
  @testset "Basic" begin
    graph = complete_graph(5)
    rewards = Dict(Edge(1, 2) => 121.0, Edge(1, 3) => 10.0, Edge(1, 4) => 10.0, Edge(1, 5) => 10.0, Edge(2, 3) => 121.0, Edge(2, 4) => 10.0, Edge(2, 5) => 10.0, Edge(3, 4) => 121.0, Edge(3, 5) => 10.0, Edge(4, 5) => 121.0)

    i = SpanningTreeInstance(graph, rewards)
    s = st_prim(i)
    @test s.instance == i
    @test length(s.tree) == 4
    @test Edge(1, 2) in s.tree
    @test Edge(2, 3) in s.tree
    @test Edge(3, 4) in s.tree
    @test Edge(4, 5) in s.tree
  end

  @testset "Basic" begin
    graph = complete_graph(5)
    rewards = Dict(Edge(2, 5) => 0.0, Edge(3, 5) => 0.0, Edge(4, 5) => 1.0, Edge(1, 2) => 0.0, Edge(2, 3) => 0.0, Edge(1, 4) => 0.0, Edge(2, 4) => 0.0, Edge(1, 5) => 0.0, Edge(1, 3) => 0.0, Edge(3, 4) => 0.0)

    i = SpanningTreeInstance(graph, rewards)
    s = st_prim(i)
    @test s.instance == i
    @test length(s.tree) == 4
    @test length(unique(s.tree)) == 4 # Only unique edges.
    @test Edge(4, 5) in s.tree # Only edge with nonzero cost.
  end
end

@testset "Budgeted maximum spanning tree" begin
  @testset "Basic" begin
    graph = complete_graph(3)
    costs = Dict(Edge(1, 2) => 1.0, Edge(1, 3) => 0.5, Edge(2, 3) => 3.0)
    weights = Dict(Edge(1, 2) => 0, Edge(1, 3) => 2, Edge(2, 3) => 0)

    ε = 0.0001
    budget = 1

    # Lagrangian relaxation.
    i = BudgetedSpanningTreeInstance(graph, costs, weights, budget)
    lagrangian = st_prim_budgeted_lagrangian_search(i, ε)
    @test lagrangian.λ ≈ 0.25 atol=ε
    @test lagrangian.value ≈ 3.75 atol=ε
    @test length(lagrangian.tree) == 2
    @test Edge(1, 3) in lagrangian.tree
    @test Edge(2, 3) in lagrangian.tree
    @test _budgeted_spanning_tree_compute_weight(i, lagrangian.tree) >= budget

    # Helpers.
    a = [Edge(1, 2), Edge(1, 3)]
    b = [Edge(1, 3), Edge(1, 4)]
    @test _solution_symmetric_difference_size(a, b) == 2
    res_a, res_b = _solution_symmetric_difference(a, b)
    @test res_a == [Edge(1, 2)] # Elements that are in a but not in b
    @test res_b == [Edge(1, 4)] # Elements that are in b but not in a

    # Complete approximation algorithm.
    sol = st_prim_budgeted_lagrangian_refinement(i)
    @test sol.instance == i
    s = sol.tree
    @test length(s) == 2
    @test Edge(1, 3) in s # Only important edge in this instance: the only one to have a non-zero weight.
    @test _budgeted_spanning_tree_compute_weight(i, s) >= budget
  end

  @testset "Conformity" begin
    # More advanced tests to ensure the algorithm works as expected.

    # 1.
    graph = complete_graph(2)
    r = Dict(Edge(1, 2) => 0.0)
    w = Dict(Edge(1, 2) => 5)
    i = BudgetedSpanningTreeInstance(graph, r, w, 0)
    s = st_prim_budgeted_lagrangian_refinement(i)
    @assert s.tree == [Edge(1, 2)]

    i = BudgetedSpanningTreeInstance(graph, r, w, 20)
    s = st_prim_budgeted_lagrangian_refinement(i)
    @assert s.tree == Edge{Int}[]
  end
end
