# Why this package?

Combinatorial bandits should be used when you have to make structured decisions while not knowing exactly
their cost.

For instance, you might want to determine the best route to go to work in a new city: you have no previous
information about the traffic, you just have access to a map (a graph) and you can measure the time it takes
to cross a street (the cost of each edge). There is some structure to this problem, as your route starts
from your home and ends at your workplace while following streets that exist.

To solve the problem, you will start picking some route from home to your workplace and measure the time
you need for each segment of your commute. This gives you a bit of information about each edge. Then,
the next day, you can use this information to pick a possibly better route.

Every commute is a "round", in bandit terminology: you interact with the city by following a route; in
return, you get information: the time you need for each segment. An statistically efficient algorithm will
take few rounds to find the best route.

Combinatorial bandits consider that the weights do not change over time: they always follow the same
probability distribution. They focus on learning this distribution so that they can make the best decision.

## Definitions

A combinatorial set defines what decisions you can make. An arm is one part of the decision (in the above
example, each segment in your route — an edge in the graph — is a arm).

When you use a solution, you get a reward. This package focuses on the semi-bandit paradigm, meaning that
you get a reward for each arm that you use. (The other possibility is full-bandit information, in which
case you only get the total reward: for the above example, that would be the total time for your commute.)

A policy is a way to determine the next solution you will use. They can be very simple, like Thompson
sampling, or more involved, like ESCB or OSSB. The latter two have better statistical properties than
Thompson sampling, meaning that they will require fewer rounds to get the same level of information.

Specifically in this package, policies have several implementations. Some of them are faster than others.
See details in `usage.md`.

## Difference with reinforcement learning

Combinatorial bandits are a part of reinforcement learning (RL). However, unlike usual reinforcement learning,
combinatorial bandits ignore state. This is reflected by the hypothesis that the weights have a fixed
distribution.

## Difference with stochastic optimisation

Combinatorial bandits start with no knowledge about the weights. Stochastic optimisation is about finding the
best decision based on some preexisting stochastic model.
