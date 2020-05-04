#!/usr/bin/env python
# coding: utf-8

# ### Monte Carlo with Exploring Starts for estimating an optimal Blackjack policy
# 
# This corresponds to the algorithm described in Chapter 5.3 of [_Reinforcement Learning: An Introduction_](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto

# In[66]:


# Dependencies
from typing import Any, Tuple

import gym

from rl_agents.blackjack.MonteCarloAgent import MonteCarloAgent

# ### Setup Blackjack agent and environment

# In[67]:
RANDOM_SEED = 2

env = gym.make('Blackjack-v0')
env.seed(seed=RANDOM_SEED)
agent = MonteCarloAgent(action_space=env.action_space,
                        obs_space=env.observation_space,
                        seed=RANDOM_SEED)

# In[68]:


print(agent.actions)
print(agent.policy)
# ### Setup Blackjack Experiment

# In[65]:

for i_episode in range(5):
    observation = env.reset()
    reward = 0
    done = False
    t = 0
    episode_ts = []
    while not done:
        action = agent.agent_step(reward=reward,
                                  observation=observation)

        observation, reward, done, info = env.step(action)

        episode_ts.append((action, observation, reward))

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

        t += 1

    agent.agent_end(episode_ts=episode_ts)





# In[36]:

print(agent.policy)

# In[ ]:
