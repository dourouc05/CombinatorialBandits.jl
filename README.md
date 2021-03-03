# CombinatorialBandits

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](http://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/dourouc05/CombinatorialBandits.jl.svg?branch=master)](https://travis-ci.org/dourouc05/CombinatorialBandits.jl)
[![Coverage Status](https://coveralls.io/repos/dourouc05/CombinatorialBandits.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/dourouc05/CombinatorialBandits.jl?branch=master)
[![codecov.io](http://codecov.io/github/dourouc05/CombinatorialBandits.jl/coverage.svg?branch=master)](http://codecov.io/github/dourouc05/CombinatorialBandits.jl?branch=master)

This package implements several algorithms to deal with combinatorial multi-armed bandit (CMAB), including the first polynomial-time optimum-regret algorithms: AESCB ([described in our paper](https://arxiv.org/abs/2002.07258)) and AOSSB (article in press).

See also [Bandits.jl](https://github.com/rawls238/Bandits.jl), focusing on multi-armed bandits (i.e. not combinatorial).

To install:

```julia
]add CombinatorialBandits
```

Example usage:

```julia
using CombinatorialBandits, Distributions

n = 20
m = 8
ε = 0.1
distr = Distribution[Bernoulli(.5 + ((i % 3 == 0) ? ε : -ε)) for i in 1:n]

i = MSet(distr, 8, MSetAlgosSolver())
@time simulate(i, ThompsonSampling(), 200)
@time simulate(i, LLR(), 200)
@time simulate(i, CUCB(), 200)
@time simulate(i, ESCB2(ESCB2Budgeted(.1, true)), 200)
```

## Citing

If you use this package in your research, please cite either article: 

```
@article{cuvelier2021aescb,
    author = {Cuvelier, Thibaut and Combes, Richard and Gourdin, Eric},
    title = {Statistically Efficient, Polynomial-Time Algorithms for Combinatorial Semi-Bandits},
    year = {2021},
    issue_date = {March 2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {5},
    number = {1},
    url = {https://doi.org/10.1145/3447387},
    doi = {10.1145/3447387},
    journal = {Proc. ACM Meas. Anal. Comput. Syst.},
    month = feb,
    articleno = {09},
    numpages = {31},
    keywords = {combinatorial bandits, combinatorial optimization, bandits}
}

@article{cuvelier2021glpg,
  title={Asymptotically Optimal Strategies For Combinatorial Semi-Bandits in Polynomial Time},
  author={Cuvelier, Thibaut and Combes, Richard and Gourdin, Eric},
  journal={arXiv preprint arXiv:2102.07254},
  year={2021}
}
```
