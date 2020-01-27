# CombinatorialBandits

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](http://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/dourouc05/CombinatorialBandits.jl.svg?branch=master)](https://travis-ci.org/dourouc05/CombinatorialBandits.jl)
[![Coverage Status](https://coveralls.io/repos/dourouc05/CombinatorialBandits.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/dourouc05/CombinatorialBandits.jl?branch=master)
[![codecov.io](http://codecov.io/github/dourouc05/CombinatorialBandits.jl/coverage.svg?branch=master)](http://codecov.io/github/dourouc05/CombinatorialBandits.jl?branch=master)

This package implements several algorithms to deal with combinatorial multi-armed bandit (CMAB).

See also [Bandits.jl](https://github.com/rawls238/Bandits.jl), focusing on multi-armed bandits (i.e. not combinatorial).

To install:

```julia
Pkg.add("CombinatorialBandits")
```

Example usage:

```julia
using CombinatorialBandits, Distributions

n = 20
ε = 0.1
distr = Distribution[Bernoulli(.5 + ((i % 3 == 0) ? ε : -ε)) for i in 1:n]

i = UncorrelatedPerfectBipartiteMatching(distr, MSetAlgosSolver())
@time simulate(i, ThompsonSampling(), 200)
@time simulate(i, LLR(), 200)
@time simulate(i, CUCB(), 200)
@time simulate(i, ESCB2(.1, ESCB2Budgeted(.1, true)), 200)
```
