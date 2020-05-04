from typing import Any, List, Tuple

import gym
import numpy as np

from rl_agents.abstract import IBaseAgent


class MonteCarloAgent(IBaseAgent):
    """Blackjack agent implementing the Monte Carlo with exploring starts
    algorithm for control"""

    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 obs_space: Tuple[gym.spaces.Discrete,
                                  gym.spaces.Discrete,
                                  gym.spaces.Discrete],
                 discount_factor: float = 1.0,
                 seed: int = 2) -> None:
        """Initializes the policy vector, state-action matrix, and returns
        matrix.

        TODO: Optimize memory usage by updating the mean and counts for returns
            incrementally, and using dictionaries for all data structures.

        Args:
            action_space: Discrete action space (Hit or Stay).
            obs_space: Discrete state space (the players current sum,
            the dealer's one showing card (1-10 where 1 is ace),
            and whether or not the player holds a usable ace (0 or 1))
            discount_factor: (optional) discount applied to future rewards
            seed: (optional) Set seed for random number generator
        """
        self.discount_factor = discount_factor
        self.rand_generator = np.random.RandomState(seed=seed)
        self.actions = np.arange(action_space.n)

        # To speed up training, I will apply some domain knowledge and
        # initialize the policy to STAY if the player's score is 20 or 21,
        # and HIT otherwise
        self.policy = np.ones(shape=(obs_space[0].n,  # Player's score
                                     obs_space[1].n,  # Dealers's Card
                                     obs_space[2].n),  # has_usable_ace
                              dtype=int)
        for i in range(20, 22):
            self.policy[i] = np.zeros(shape=(
                obs_space[1].n,  # Dealers's Card
                obs_space[2].n),  # has_usable_ace
            )

        self.action_values = np.zeros(shape=(obs_space[0].n,  # Player's score
                                             obs_space[1].n,  # Dealers's Card
                                             obs_space[2].n,  # has_usable_ace
                                             action_space.n),  # {STAY, HIT}
                                      dtype=float)

        # TODO: This is pretty absurd... Refactor as soon as possible.
        self.returns = [[[[[] for i in range(action_space.n)]  # {STAY, HIT}
                         for i in range(obs_space[2].n)]  # has_usable_ace
                         for i in range(obs_space[1].n)]  # Dealers's Card
                        for i in range(obs_space[0].n)]  # Player's score

    @staticmethod
    def bool2int(boolean: bool) -> int:
        """Converts boolean to an integer value (0 or 1).

        Args:
            boolean: Boolean variable

        Returns:
            Integer representing the boolean
        """
        if boolean:
            return 1
        else:
            return 0

    def _argmax(self,
                vector: np.ndarray) -> int:
        """Select the index containing the max value of the vector (with ties
        broken arbitrarily)

        Returns:
            Integer representing the maximizing action
        """
        ties = []
        max_val = -np.inf
        for i in range(len(vector)):
            if vector[i] > max_val:
                max_val = vector[i]
                ties = []

            if vector[i] == max_val:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def _select_action(self,
                       score: int,
                       dealer_card: int,
                       has_usable_ace: int) -> int:
        """Select action according to agent's policy.

        Args:
            score: Current sum of all cards in player's hand
            dealer_card: Value of card shown by the dealer
            has_usable_ace: Whether of not the player holds a usable ace (0
            or 1)

        Returns:
            Integer representing the action chosen according to self.policy
        """

        return self.policy[score, dealer_card, has_usable_ace]

    def agent_start(self,
                    observation: Tuple[Any]) -> int:
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Tuple): the state observation from the
            environment's reset() function.
        Returns:
            The first action the agent takes.
        """
        pass

    def agent_step(self,
                   reward: float,
                   observation: Tuple[Any]) -> int:
        """A step taken by the agent.

        Args:
            reward: Reward received for taking the last action taken
            observation: State observation from the environment's
            step based, where the agent ended up after the last step
        Returns:
            The action the agent is taking.
        """
        # NB: We don't do anything with the reward because for Monte Carlo
        # learning we update our action_values at the end of an episode.
        return self._select_action(score=observation[0],
                                   dealer_card=observation[1],
                                   has_usable_ace=self.bool2int(observation[2]))

    def agent_end(self,
                  episode_ts: List[Tuple[float, Tuple[int, int, bool], float]]) -> None:
        """Run when the agent terminates.

        For the Monte Carlo agent, we update our action-values at the end of
        each episode. This update occurs here.

        Args:
            episode_ts: Time-series list of (action, observation, reward)
            tuples, sorted earliest to latest.
        """
        # initialize return and reverse time-series
        return_val = 0
        episode_ts.reverse()

        # For each step of the episode, t = T-1, T-2, ..., 0
        for timestep in episode_ts:
            # Extract time step information
            action, observation, reward = timestep
            score, dealer_card, has_usable_ace = observation
            has_usable_ace = self.bool2int(has_usable_ace)

            # Update discounted return_val
            return_val = (self.discount_factor * return_val) + reward

            # Append to state-action returns list
            self.returns[score][dealer_card][has_usable_ace][action].append(return_val)

            self.action_values[
                score,
                dealer_card,
                has_usable_ace,
                action] = np.mean(self.returns[score][dealer_card][has_usable_ace][action])

            self.policy[
                score,
                dealer_card,
                has_usable_ace
            ] = self._argmax(self.action_values[score,
                                                dealer_card,
                                                has_usable_ace])

    def agent_cleanup(self) -> None:
        """Cleanup done after the agent ends."""
        pass
