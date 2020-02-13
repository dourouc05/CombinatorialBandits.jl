using Test

@testset "m-sets" begin
  @testset "Interface" begin
    @test_throws ErrorException MSetInstance(Float64[5, 4, 3], -1)
    @test_throws ErrorException MSetInstance(Float64[5, 4, 3], 0)
  end

  @testset "Basic" begin
    m = 2
    i = MSetInstance(Float64[5, 4, 3], m)
    g = msets_greedy(i)
    d = msets_dp(i)
    l = ! is_travis && msets_lp(i, solver=Gurobi.Optimizer)

    @test i.m == m

    @test value(g) == value(d)
    @test length(g.items) <= m
    @test length(d.items) <= m

    @test g.instance == i
    @test d.instance == i

    if ! is_travis
      @test value(g) == value(l)
      @test length(l.items) <= m
      @test l.instance == i
    end
  end
end

@testset "Budgeted m-sets" begin
  @testset "Interface" begin
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4, 3], Int[1, 1, 1], -1)
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4, 3], Int[1, 1, 1], 0)
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4, 3], Int[1, 1, 1], 2, budget=-1)
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4, 3], Int[1, -1, 1], 2)
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4, 3], Int[1, 1], 0)
    @test_throws ErrorException BudgetedMSetInstance(Float64[5, 4], Int[1, 1, 1], 0)
  end

  function test_solution_at(s::BudgetedMSetSolution, kv::Dict{Int, Float64})
    for (k, v) in kv
      @test value(s, k) ≈ v
    end
  end

  function test_items_at(s::BudgetedMSetSolution, kv::Dict{Int, Vector{Int}})
    for (k, v) in kv
      @test items(s, k) == v
    end
  end

  @testset "Basic" begin
    m = 2
    i = BudgetedMSetInstance(Float64[5, 4, 3], Int[1, 1, 1], m)
    d = budgeted_msets_dp(i)
    l = ! is_travis && budgeted_msets_lp_all(i, solver=Gurobi.Optimizer)

    @test i.m == m
    @test d.instance == i

    @test d.state[m, 0 + 1, 0 + 1] == 9.0
    @test d.state[m, 0 + 1, 1 + 1] == 9.0
    @test d.state[m, 0 + 1, 2 + 1] == 9.0
    @test d.state[m, 0 + 1, 3 + 1] == -Inf

    @test d.solutions[m, 0, 0] == [1, 2]
    @test d.solutions[m, 0, 1] == [1, 2]
    @test d.solutions[m, 0, 2] == [1, 2]
    @test d.solutions[m, 0, 3] == [-1]

    if ! is_travis
      for i in 0:3
        @test l.state[m, 0 + 1, i + 1] ≈ d.state[m, 0 + 1, i + 1]
        @test l.solutions[m, 0, i] == d.solutions[m, 0, i]
      end
    end

    # Accessors.
    expected_items = Dict{Int, Vector{Int}}(0 => [1, 2], 1 => [1, 2], 2 => [1, 2], 3 => [-1])
    test_items_at(d, expected_items)
    ! is_travis && test_items_at(l, expected_items)

    expected = Dict{Int, Float64}(0 => 9.0, 1 => 9.0, 2 => 9.0, 3 => -Inf)
    test_solution_at(d, expected)
    ! is_travis && test_solution_at(l, expected)
  end

  @testset "Conformity" begin
    # More advanced tests to ensure the algorithm works as expected.

    # 1
    a = 3.612916190062782
    b = 7.225832380125564
    v = Float64[a, a, b, b, b, b, b, b, b, b]
    w = Int[32, 32, 32, 32, 0, 32, 32, 32, 0, 32]
    m = 3

    i = BudgetedMSetInstance(v, w, m)
    d = budgeted_msets_dp(i)
    # TODO: stop the algorithm in this case? Don't waste too much time on this part of the table.
    l = ! is_travis && budgeted_msets_lp_select(i, [0, 96, 97, 320], solver=Gurobi.Optimizer)
    expected = Dict{Int, Float64}(0 => 3 * b, 96 => 3 * b, 96 + 1 => -Inf, 320 => -Inf) # No more solutions after 96.
    test_solution_at(d, expected)
    ! is_travis && test_solution_at(l, expected)

    # 2
    v = [7.840854066284411, 3.9204270331422055, 7.840854066284411, 3.9204270331422055, 7.840854066284411, 7.840854066284411, 7.840854066284411, 7.840854066284411, 15.681708132568822, 5.227236044189607]
    w = [16, 32, 16, 24, 16, 16, 0, 16, 0, 32]
    m = 3

    i = BudgetedMSetInstance(v, w, m)
    d = budgeted_msets_dp(i)
    l = ! is_travis && budgeted_msets_lp_select(i, [0, 4, 20, 24, 25, 32, 33, 40, 41, 48, 49, 72, 73, 96, 97, 280, 319, 320], solver=Gurobi.Optimizer)

    a = 31.363416265137644
    b = 28.74979824304284
    c = 24.829371209900632
    e = 16.988517143616225
    expected = Dict{Int, Float64}(0 => a, 4 => a, 20 => a, 24 => a, 25 => a, 32 => a,
      33 => b, 40 => b, 41 => b, 48 => b,
      49 => c, 72 => e, 73 => e,
      96 => -Inf, 97 => -Inf, 280 => -Inf, 319 => -Inf, 320 => -Inf
    )
    test_solution_at(d, expected)
    ! is_travis && test_solution_at(l, expected)
  end
end
