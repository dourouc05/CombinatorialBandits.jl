# How to use this package?

First, settle on a combinatorial problem to solve, like `MSet`.

Second, import the required packages. These include CombinatorialBandits,
for obvious reasons, but also Distributions.

    using CombinatorialBandits, Distributions

Third, define an instance of this problem to solve. The instance is mostly
characterised by the probability distributions of the reward. For instance,
take an m-set with Bernoulli rewards (even arms have a winning probability
of 80%, the others 20%). Decide the solver to use (like `MSetAlgosSolver`):
the possible solvers depend on the combinatorial problem to solve.

    n = 10
    distr = Distribution[Bernoulli(i % 2 == 0 ? .8 : .2) for i in 1:n]
    i = MSet(distr, m, MSetAlgosSolver())

Fourth, start your combinatorial bandit experiment for a given number of
iterations, like 1000:

    n_iter = 1000
    end_state, trace = simulate(i, ThompsonSampling(), n_iter, with_trace=true)
