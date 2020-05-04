from typing import Any, Tuple

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
                                  gym.spaces.Discrete]) -> None:
        """Initializes the policy vector, state-action matrix, and returns
        matrix.

        TODO: Optimize memory usage by updating the mean and counts for returns
            incrementally, and using dictionaries for all data structures.

        Args:
            action_space (spaces.Discrete): Discrete action space (Hit or Stay).
            obs_space (Tuple): Discrete state space (the players current sum,
            the dealer's one showing card (1-10 where 1 is ace),
            and whether or not the player holds a usable ace (0 or 1))
        """
        self.actions = np.arange(action_space.n)
        self.policy = np.zeros(shape=(obs_space[0].n,  # Player's score
                                      obs_space[1].n,  # Dealers's Card
                                      obs_space[2].n),  # has_usable_ace
                               dtype=int)

        # To speed up training, I will apply some domain knowledge and
        # initialize the policy to STAY if the player's score is 20 or 21.
        for i in range(20, 22):
            self.policy[i] = np.ones(shape=(
                obs_space[1].n,  # Dealers's Card
                obs_space[2].n),  # has_usable_ace
            )

        self.action_values = np.zeros(shape=(obs_space[0].n,  # Player's score
                                             obs_space[1].n,  # Dealers's Card
                                             obs_space[2].n,  # has_usable_ace
                                             action_space.n),  # {STAY, HIT}
                                      dtype=float)
        self.rewards = np.zeros(shape=(obs_space[0].n,  # Player's score
                                       obs_space[1].n,  # Dealers's Card
                                       obs_space[2].n,  # has_usable_ace
                                       action_space.n),  # {STAY, HIT}
                                dtype=float)

    def bool2int(self,
                 boolean: bool) -> int:
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
            reward (float): the reward received for taking the last action taken
            observation (Tuple): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        return self._select_action(score=observation[0],
                                   dealer_card=observation[1],
                                   has_usable_ace=bool2int(observation[2])))

    def agent_end(self,
                  reward: float) -> None:
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
            terminal state.
        """
        pass

    def agent_cleanup(self) -> None:
        """Cleanup done after the agent ends."""
        pass
