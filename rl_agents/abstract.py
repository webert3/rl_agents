"""An abstract class that specifies the Agent API for the Open AI Gym.

An abstract class that specifies the Agent API for the Open AI Gym. Based on
Agent API compatible with the RL-Glue library:
(https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue)
"""

from __future__ import print_function
from typing import Any, Tuple

from abc import abstractmethod


class IBaseAgent:
    """Implements the agent for the Open AI Gym environments"""

    def __init__(self):
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def agent_end(self,
                  reward: float) -> None:
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
            terminal state.
        """
        pass

    @abstractmethod
    def agent_cleanup(self) -> None:
        """Cleanup done after the agent ends."""
        pass
