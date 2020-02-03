using Test

@testset "Maximum bipartite matching" begin
  @testset "Interface" begin
    # Error here: edges to the same side of the bipartite graph.
    graph = complete_bipartite_graph(5, 5)
    rewards = Dict{Edge{Int}, Float64}()
    weights = Dict{Edge{Int}, Int}()
    for i in 1:5
      for j in 1:5
        rewards[Edge(i, j)] = (i == j) ? 1.0 : 0.0
        weights[Edge(i, j)] = (i == j) ? 0 : 1
      end
    end
    budget = 0
    ε = 0.0001
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    @test_throws AssertionError matching_hungarian_budgeted_lagrangian_search(i, ε)
  end

  @testset "Basic" begin
    # Three nodes on each side (1-2-3 to the left, 4-5-6 to the right),
    # optimum is matching them in order (1-4, 2-5, 3-6)
    graph = complete_bipartite_graph(3, 3)
    rewards = Dict(Edge(1, 4) => 1.0, Edge(1, 5) => 0.0, Edge(1, 6) => 0.0, Edge(2, 4) => 0.0, Edge(2, 5) => 1.0, Edge(2, 6) => 0.0, Edge(3, 4) => 0.0, Edge(3, 5) => 0.0, Edge(3, 6) => 1.0)

    i = BipartiteMatchingInstance(graph, rewards)
    s = matching_hungarian(i)
    @test s.instance == i
    @test length(s.solution) == 3
    @test Edge(1, 4) in s.solution
    @test Edge(2, 5) in s.solution
    @test Edge(3, 6) in s.solution
    @test s.value ≈ 3.0
  end
end

@testset "Budgeted maximum bipartite matching" begin
  @testset "Interface" begin
    graph = complete_graph(3)
    rewards = Dict(Edge(1, 2) => 1.0, Edge(1, 3) => 0.5, Edge(2, 3) => 3.0)
    weights = Dict(Edge(1, 2) => 0, Edge(1, 3) => 2, Edge(2, 3) => 0)
    @test_throws ErrorException BudgetedBipartiteMatchingInstance(graph, rewards, weights, 0)

    graph = complete_bipartite_graph(2, 2)
    rewards = Dict(Edge(1, 3) => 1.0, Edge(1, 4) => 0.0, Edge(2, 3) => 0.0, Edge(2, 4) => 1.0)
    weights = Dict(Edge(1, 3) => 0, Edge(1, 4) => 1, Edge(2, 3) => 1, Edge(2, 4) => 0)
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, 0)

    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁻=1.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁻=2.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁺=1.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁺=0.2)
  end

  @testset "Basic" begin
    # Two nodes on each side (1-2 to the left, 3-4 to the right),
    # unbudgeted optimum is matching them in order (1-3, 2-4).
    # Two solutions: 1-3 and 2-4 (reward: 2; weight: 0) or 1-4 and 2-3 (reward: 0; weight: 2).
    graph = complete_bipartite_graph(2, 2)
    rewards = Dict(Edge(1, 3) => 1.0, Edge(1, 4) => 0.0, Edge(2, 3) => 0.0, Edge(2, 4) => 1.0)
    weights = Dict(Edge(1, 3) => 0, Edge(1, 4) => 1, Edge(2, 3) => 1, Edge(2, 4) => 0)
    ε = 0.0001

    # No budget constraint.
    budget = 0
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 3) in s.solution
    @test Edge(2, 4) in s.solution
    @test s.value ≈ 2.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 2.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 3) in s.solution
    @test Edge(2, 4) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 2.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 3) in s.solution
    @test Edge(2, 4) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 2.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    # Loose budget constraint.
    budget = 1
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test s.value ≈ 1.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    # Tight budget constraint.
    budget = 2
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test s.value ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test length(s.solution) == 2
    @test Edge(1, 4) in s.solution
    @test Edge(2, 3) in s.solution
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε

    # Infeasible, but the Lagrangian relaxation does not really care.
    budget = 3
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 2

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 0

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test length(s.solution) == 0
  end

  @testset "Conformity" begin
    # Three nodes on each side (1-2-3 to the left, 4-5-6 to the right),
    # optimum is matching them in order (1-4, 2-5, 3-6)
    graph = complete_bipartite_graph(3, 3)
    rewards = Dict(Edge(1, 4) => 1.0, Edge(1, 5) => 0.0, Edge(1, 6) => 0.0, Edge(2, 4) => 0.0, Edge(2, 5) => 1.0, Edge(2, 6) => 0.0, Edge(3, 4) => 0.0, Edge(3, 5) => 0.0, Edge(3, 6) => 1.0)
    weights = Dict(Edge(1, 4) => 0, Edge(1, 5) => 1, Edge(1, 6) => 1, Edge(2, 4) => 1, Edge(2, 5) => 0, Edge(2, 6) => 1, Edge(3, 4) => 1, Edge(3, 5) => 1, Edge(3, 6) => 0)
    ε = 0.0001

    budget = 0
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 3
    @test Edge(1, 4) in s.solution
    @test Edge(2, 5) in s.solution
    @test Edge(3, 6) in s.solution
    @test s.value ≈ 3.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 3.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    budget = 2
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 3
    @test ! (Edge(1, 4) in s.solution)
    @test ! (Edge(2, 5) in s.solution)
    @test ! (Edge(3, 6) in s.solution)
    @test s.value ≈ 1.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 3.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 3
    @test ! (Edge(1, 4) in s.solution)
    @test ! (Edge(2, 5) in s.solution)
    @test ! (Edge(3, 6) in s.solution)
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 3.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test length(s.solution) == 3
    @test ! (Edge(1, 4) in s.solution)
    @test ! (Edge(2, 5) in s.solution)
    @test ! (Edge(3, 6) in s.solution)
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 3.0 atol=ε

    # Larger graph, to test more parts of the approximation scheme
    graph = complete_bipartite_graph(5, 5)
    rewards = Dict{Edge{Int}, Float64}()
    weights = Dict{Edge{Int}, Int}()
    for i in 1:5
      for j in 1:5
        k = j + 5
        rewards[Edge(i, k)] = (i == j) ? 1.0 : 0.0
        weights[Edge(i, k)] = (i == j) ? 0 : 1
      end
    end
    ε = 0.0001

    budget = 0
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)
    s = matching_hungarian_budgeted_lagrangian_search(i, ε)

    @test s.instance == i
    @test length(s.solution) == 5
    @test Edge(1, 6) in s.solution
    @test Edge(2, 7) in s.solution
    @test Edge(3, 8) in s.solution
    @test Edge(4, 9) in s.solution
    @test Edge(5, 10) in s.solution
    @test s.value ≈ 5.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 5.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_refinement(i)

    @test s.instance == i
    @test length(s.solution) == 5
    @test Edge(1, 6) in s.solution
    @test Edge(2, 7) in s.solution
    @test Edge(3, 8) in s.solution
    @test Edge(4, 9) in s.solution
    @test Edge(5, 10) in s.solution
    @test s.value ≈ 5.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 5.0 atol=ε
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε

    s = matching_hungarian_budgeted_lagrangian_approx_half(i)

    @test s.instance == i
    @test s.value >= 2.5
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) >= 2.5
    @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) >= 0.0
  end
end
