"""Agent for the Mountain Car Task"""
from typing import Any, Tuple

from rl_agents.abstract import IBaseAgent


class MountainCar(IBaseAgent):
    """Implements the agent for the Open AI Gym environments"""

    def __init__(self):
        pass

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
