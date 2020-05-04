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
        self.policy = np.zeros(shape=(obs_space[0].n,  # Player's score
                                      obs_space[1].n,  # Dealers's Card
                                      obs_space[2].n),  # has_usable_ace
                               dtype=int)
        self.action_values = np.zeros(shape=(obs_space[0].n,  # Player's score
                                             obs_space[1].n,  # Dealers's Card
                                             obs_space[2].n,  # has_usable_ace
                                             action_space.n),  # {HIT, STAY}
                                      dtype=float)
        self.rewards = np.zeros(shape=(obs_space[0].n,  # Player's score
                                       obs_space[1].n,  # Dealers's Card
                                       obs_space[2].n,  # has_usable_ace
                                       action_space.n),  # {HIT, STAY}
                                dtype=float)

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
        pass

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
