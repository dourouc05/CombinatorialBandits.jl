# Choose the underlying optimisation solver.
# opt_solver = CPLEX.Optimizer
opt_solver = Gurobi.Optimizer

function build_gl_solver(gl_solver_sym::Symbol, λ::Float64, η::Float64, n_iter::Int, solve_all_budgets_at_once::Bool)
  if gl_solver_sym == :budgeted
    return OSSBSubgradientBudgeted(CombinatorialBandits.OSSBSubgradientBudgetedFixedParameterChoice(λ, η), n_iter, solve_all_budgets_at_once) 
  elseif gl_solver_sym == :exact
    return OSSBExact()
  elseif gl_solver_sym == :exact_naive
    return OSSBExactNaive(opt_solver)
  elseif gl_solver_sym == :exact_smart
    return OSSBExactSmart()
  else
    error("Unknown option: gl_solver_sym = $(gl_solver_sym)")
  end
end

@testset "Graves-Lai" begin
  @testset "m-sets in dimension $d" for d in [2, 3, 4]
    @testset "Solver: $gl_solver_sym" for gl_solver_sym in [:exact, :exact_smart, :exact_naive, :budgeted]
      @testset "m = $m" for m in 1:floor(Int, d/2)
        # Instance parameters
        θ = Dict{Int, Float64}()
        for i in 1:m
          θ[i] = 2
        end
        for i in (m + 1):d
          θ[i] = 1
        end
        distr = Distribution[Normal(p, 1/2) for p in values(θ)]

        # Compute the theoretical value.
        # TODO: Move this to a new GL solver (the original paper should have the corresponding solution?).
        rewards = sort(collect(values(θ)), rev=true)
        @show rewards
        th_val = sum(1.0 / (rewards[m] - rewards[i]) for i in (m + 1):d)

        # Determine the parameters for this problem.
        λ = 1000.0
        η = 0.5
        n_iter = 15
        
        if m == 1 # d ∈ [2, 3, 4]
          λ = 1.0
        elseif d == 4 && m == 2
          λ = 0.2
          η = 0.2
          n_iter = 200
        end
        
        # Start the approximation algorithm.
        s = MSetLPSolver(get_solver())
        i = MSet(distr, m, s)
        gl_solver = build_gl_solver(gl_solver_sym, λ, η, n_iter, true) 
        gl = CombinatorialBandits.optimise_graves_lai(i, gl_solver, θ, with_trace=false)
        if gl_solver isa OSSBSubgradientBudgeted
          @test gl.objective ≈ th_val atol=1.0e-0
        else
          @test gl.objective ≈ th_val atol=1.0e-4
        end
      end
    end
  end

  bipartite_target_values = Dict{Int, Float64}(2 => 1.0, 3 => 3.0, 4 => 6.0, 5 => 27.8697)
  # Starting at 4, too naïve solvers take really a lot of time.

  @testset "Bipartite matchings in dimension $n" for n in [2, 3]
    @testset "Solver: $gl_solver_sym" for gl_solver_sym in [:exact, :exact_smart, :exact_naive, :budgeted]
      # Instance parameters
      graph = complete_bipartite_graph(n, n)
      θ = Dict((i, j) => (i == j) ? 2.0 : 1.0 for i in 1:n, j in 1:n)
      distr = Distribution[Normal(θ[i, j], 1/2) for i in 1:n, j in 1:n]

      # Determine the parameters for this problem.
      if n == 2
        λ = 1.0
        η = 0.05
        n_iter = 200
      elseif n == 3
        λ = 0.1
        η = 0.2
        n_iter = 400
      end
      
      # Start the approximation algorithm.
      s = PerfectBipartiteMatchingLPSolver(get_solver())
      i = UncorrelatedPerfectBipartiteMatching(distr, s)
      gl_solver = build_gl_solver(gl_solver_sym, λ, η, n_iter, true) 
      gl = CombinatorialBandits.optimise_graves_lai(i, gl_solver, θ, with_trace=false)
      if gl_solver isa OSSBSubgradientBudgeted
        @test gl.objective ≈ bipartite_target_values[n] atol=1.0e-0
      else
        @test gl.objective ≈ bipartite_target_values[n] atol=1.0e-4
      end
    end
  end
end
