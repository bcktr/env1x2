Env1x2 - Gymnasium Environment for La Quiniela

The goal is to select strategic actions based on combinations of triples and doubles to maximize rewards while considering costs, reductions, and disproportionately high rewards for exceptional accuracy.


Action Space: Discrete(6)
The agent selects a betting strategy (predefined in the strategy attribute).

Observation Space: Box(low=0.0, high=1.0, shape=(14, 3))
Each state represents the probabilities of outcomes (1, X, 2) for 14 matches.

Rewards are calculated based on the number of correct predictions and the prize associated with real-world data.

For questions or suggestions, please contact the developer or open an Issue in this repository.