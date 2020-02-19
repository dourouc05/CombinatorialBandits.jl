using Test

@testset "Maximum bipartite matching" begin
  @testset "Interface" begin
    n = 5

    # Error: the graph is not bipartite.
    graph = complete_graph(n) # Requires n > 2
    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:n
        if i != j
          rewards[Edge(i, j)] = 1.0
        end
      end
    end
    @test_throws ErrorException BipartiteMatchingInstance(graph, rewards)

    # Error: edges to the same side of the bipartite graph.
    # Both in the left part.
    graph = complete_bipartite_graph(n, n)
    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:n
        rewards[Edge(i, j)] = 1.0
      end
    end
    i = BipartiteMatchingInstance(graph, rewards)
    @test_throws ErrorException matching_hungarian(i)

    # Both in the right part.
    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:n
        rewards[Edge(i + n, j + n)] = 1.0
      end
    end
    i = BipartiteMatchingInstance(graph, rewards)
    @test_throws ErrorException matching_hungarian(i)

    # Automatic handling: edges are not from V1 to V2 in the bipartite graph.
    rewards = Dict{Edge{Int}, Float64}()
    weights = Dict{Edge{Int}, Int}()
    for i in 1:n
      for j in 1:n
        rewards[Edge(j, i + n)] = 1.0
      end
    end
    i = BipartiteMatchingInstance(graph, rewards)
    @test matching_hungarian(i) != π # Just a @test_nothrows: https://github.com/JuliaLang/julia/issues/18780

    # Test the partial-copy constructor.
    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:n
        rewards[Edge(j, i + n)] = 2.0
      end
    end
    i2 = BipartiteMatchingInstance(i, rewards)
    @test i.graph == i2.graph
    @test i.reward != i2.reward # Only difference to be found.
    @test i2.reward == rewards
    @test i.partition == i2.partition
    @test i.n_left == i2.n_left
    @test i.n_right == i2.n_right
    @test i.vertex_left == i2.vertex_left
    @test i.vertex_right == i2.vertex_right
  end

  @testset "Basic: 3×3" begin
    # Three nodes on each side (1-2-3 to the left, 4-5-6 to the right),
    # optimum is matching them in order (1-4, 2-5, 3-6)
    graph = complete_bipartite_graph(3, 3)
    rewards = Dict(Edge(1, 4) => 1.0, Edge(1, 5) => 0.0, Edge(1, 6) => 0.0, Edge(2, 4) => 0.0, Edge(2, 5) => 1.0, Edge(2, 6) => 0.0, Edge(3, 4) => 0.0, Edge(3, 5) => 0.0, Edge(3, 6) => 1.0)

    i = BipartiteMatchingInstance(graph, rewards)

    s1 = matching_hungarian(i)
    s2 = matching_dp_imperfect(i)
    s3 = matching_dp(i)

    for s in [s1, s2, s3]
      @test s.instance == i
      @test length(s.solution) == 3
      @test Edge(1, 4) in s.solution
      @test Edge(2, 5) in s.solution
      @test Edge(3, 6) in s.solution
      @test s.value ≈ 3.0
    end
  end

  @testset "Conformity: 5×5" begin
    # Just a larger graph. Size 5 is what is used for conformity tests in budgeted matchings.
    n = 5
    graph = complete_bipartite_graph(n, n)

    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:n
        rewards[Edge(i, j + n)] = 1.0
      end
    end

    i = BipartiteMatchingInstance(graph, rewards)

    s1 = matching_hungarian(i)
    s2 = matching_dp_imperfect(i)
    s3 = matching_dp(i)

    for s in [s1, s2, s3]
      @test s.instance == i
      @test length(s.solution) == n
      for i in 1:n
        @test Edge(i, i + n) in s.solution
      end
      @test s.value ≈ n
    end

    # Ensure one node has no matching in V1.
    graph = complete_bipartite_graph(n - 1, n)

    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:(n - 1)
      for j in 1:n
        rewards[Edge(i, j + n - 1)] = 1.0
      end
    end

    i = BipartiteMatchingInstance(graph, rewards)

    s1 = matching_hungarian(i)
    s2 = matching_dp_imperfect(i)
    s3 = matching_dp(i)

    for s in [s1, s2, s3]
      @test s.instance == i
      @test length(s.solution) == n - 1
      @test s.value ≈ n - 1.0
    end

    # Ensure one node has no matching in V2.
    graph = complete_bipartite_graph(n, n - 1)

    rewards = Dict{Edge{Int}, Float64}()
    for i in 1:n
      for j in 1:(n - 1)
        rewards[Edge(i, j + n)] = 1.0
      end
    end

    i = BipartiteMatchingInstance(graph, rewards)

    s1 = matching_hungarian(i)
    s2 = matching_dp_imperfect(i)
    s3 = matching_dp(i)

    for s in [s1, s2, s3]
      @test s.instance == i
      @test length(s.solution) == n - 1
      @test s.value ≈ n - 1.0
    end
  end
end

@testset "Budgeted maximum bipartite matching" begin
  @testset "Interface" begin
    # Graph must be bipartite.
    graph = complete_graph(3)
    rewards = Dict(Edge(1, 2) => 1.0, Edge(1, 3) => 0.5, Edge(2, 3) => 3.0)
    weights = Dict(Edge(1, 2) => 0, Edge(1, 3) => 2, Edge(2, 3) => 0)
    @test_throws ErrorException BudgetedBipartiteMatchingInstance(graph, rewards, weights, 0)

    # Values of ζ.
    graph = complete_bipartite_graph(2, 2)
    rewards = Dict(Edge(1, 3) => 1.0, Edge(1, 4) => 0.0, Edge(2, 3) => 0.0, Edge(2, 4) => 1.0)
    weights = Dict(Edge(1, 3) => 0, Edge(1, 4) => 1, Edge(2, 3) => 1, Edge(2, 4) => 0)
    i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, 0)

    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁻=1.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁻=2.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁺=1.0)
    @test_throws ErrorException matching_hungarian_budgeted_lagrangian_refinement(i, ζ⁺=0.2)

    # Infeasible solutions.
    s1 = SimpleBudgetedBipartiteMatchingSolution(i) == SimpleBudgetedBipartiteMatchingSolution(i, edgetype(graph)[])
  end

  @testset "Helpers" begin
    @test CombinatorialBandits._edge_any_end_match(Edge(1, 2), Edge(1, 2))
    @test CombinatorialBandits._edge_any_end_match(Edge(1, 2), Edge(1, 3))
    @test CombinatorialBandits._edge_any_end_match(Edge(1, 2), Edge(4, 2))
    @test CombinatorialBandits._edge_any_end_match(Edge(1, 3), Edge(1, 2))
    @test CombinatorialBandits._edge_any_end_match(Edge(4, 2), Edge(1, 2))
    @test ! CombinatorialBandits._edge_any_end_match(Edge(1, 2), Edge(3, 4))
  end

  @testset "Basic: 2×2" begin
    # Two nodes on each side (1-2 to the left, 3-4 to the right),
    # unbudgeted optimum is matching them in order (1-3, 2-4).
    # Two solutions: 1-3 and 2-4 (reward: 2; weight: 0) or 1-4 and 2-3 (reward: 0; weight: 2).
    graph = complete_bipartite_graph(2, 2)
    rewards = Dict(Edge(1, 3) => 1.0, Edge(1, 4) => 0.0, Edge(2, 3) => 0.0, Edge(2, 4) => 1.0)
    weights = Dict(Edge(1, 3) => 0, Edge(1, 4) => 1, Edge(2, 3) => 1, Edge(2, 4) => 0)
    ε = 0.0001

    @testset "No budget constraint" begin
      # TODO: for a zero budget, always rely on standard matching algorithm? Could be much more efficient.
      budget = 0
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 2.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 2
        @test Edge(1, 3) in s.solution
        @test Edge(2, 4) in s.solution
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 2.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε
      end
    end

    @testset "Loose budget constraint" begin
      budget = 1 # The only solution respecting this budget has a total weight of 2.
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 1.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 2
        @test Edge(1, 4) in s.solution
        @test Edge(2, 3) in s.solution
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε
      end
    end

    @testset "Tight budget constraint" begin
      budget = 2
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 0.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 2
        @test Edge(1, 4) in s.solution
        @test Edge(2, 3) in s.solution
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε
      end
    end

    @testset "Impossibly tight budget constraint" begin
      budget = 3 # Infeasible, but the Lagrangian relaxation does not really care.
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value <= 0.0 # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
      end

      # The Lagrangian relaxation does not care.
      @test s1.instance == i
      @test length(s1.solution) == 2

      # The other approximation algorithms correctly return an infeasible solution.
      for s in [s2, s3, s4]
        @test length(s.solution) == 0
        @test s.value == -Inf
      end
    end
  end

  @testset "Conformity: 3×3" begin
    # Three nodes on each side (1-2-3 to the left, 4-5-6 to the right),
    # optimum is matching them in order (1-4, 2-5, 3-6)
    graph = complete_bipartite_graph(3, 3)
    rewards = Dict{Edge{Int}, Float64}()
    weights = Dict{Edge{Int}, Int}()
    for i in 1:3
      for j in 1:3
        k = j + 3
        rewards[Edge(i, k)] = (i == j) ? 1.0 : 0.0
        weights[Edge(i, k)] = (i == j) ? 0 : 1
      end
    end
    ε = 0.0001

    @testset "No budget contraint" begin
      budget = 0
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 3.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 3
        @test Edge(1, 4) in s.solution
        @test Edge(2, 5) in s.solution
        @test Edge(3, 6) in s.solution
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 3.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε
      end
    end

    @testset "Loose budget constraint" begin
      budget = 2
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 1.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 3
        @test ! (Edge(1, 4) in s.solution)
        @test ! (Edge(2, 5) in s.solution)
        @test ! (Edge(3, 6) in s.solution)
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 3.0 atol=ε
      end
    end
  end

  # Not really useful to test 4×4, as everything would be fixed by the major loops in the 1/2 scheme.

  @testset "Conformity: 5×5" begin
    # Larger graph, to test more parts of the approximation scheme (fixing four edges in the 1/2 scheme, so take
    # at least one more so that the underlying additive approximation scheme has some work to do).
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
    max_reward_edges = [Edge(1, 6), Edge(2, 7), Edge(3, 8), Edge(4, 9), Edge(5, 10)]

    @testset "No budget constraint" begin
      budget = 0
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      s1 = matching_hungarian_budgeted_lagrangian_search(i, ε)
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      @test s1.value ≈ 5.0 atol=ε # Lagrangian value.

      for s in [s1, s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 5

        for e in max_reward_edges
          @test e in s.solution
        end

        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 5.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 0.0 atol=ε
      end
    end

    @testset "Loose budget constraint" begin
      budget = 1
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      # Don't test Lagrangian relaxation now, constraint is not respected.
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      for s in [s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 5

        n_max_reward_edges = sum(e in s.solution for e in max_reward_edges)
        @test n_max_reward_edges <= 4

        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 3.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 2.0 atol=ε
      end
    end

    @testset "Tight budget constraint" begin
      budget = 5
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      # Don't test Lagrangian relaxation now, constraint is not respected.
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      for s in [s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 5

        for e in max_reward_edges
          @test ! (e in s.solution)
        end

        @test CombinatorialBandits._budgeted_bipartite_matching_compute_value(i, s.solution) ≈ 0.0 atol=ε
        @test CombinatorialBandits._budgeted_bipartite_matching_compute_weight(i, s.solution) ≈ 5.0 atol=ε
      end
    end

    @testset "Impossibly tight budget constraint" begin
      budget = 6
      i = BudgetedBipartiteMatchingInstance(graph, rewards, weights, budget)

      # Don't test Lagrangian relaxation now, constraint is not respected.
      s2 = matching_hungarian_budgeted_lagrangian_refinement(i)
      s3 = matching_hungarian_budgeted_lagrangian_approx_half(i)
      s4 = matching_dp_budgeted(i)

      for s in [s2, s3, s4]
        @test s.instance == i
        @test length(s.solution) == 0
      end
    end
  end
end
