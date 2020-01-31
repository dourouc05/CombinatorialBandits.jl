using Test

@testset "Maximum bipartite matching" begin
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
