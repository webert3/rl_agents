#!/usr/bin/env python
# coding: utf-8

# ### Monte Carlo with Exploring Starts for estimating an optimal Blackjack policy
# 
# This corresponds to the algorithm described in Chapter 5.3 of [_Reinforcement Learning: An Introduction_](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto

# In[66]:


# Dependencies
from typing import Any, Tuple

import gym

from rl_agents.abstract import IBaseAgent
from rl_agents.blackjack.MonteCarloAgent import MonteCarloAgent

# ### Setup Blackjack agent and environment

# In[67]:


env = gym.make('Blackjack-v0')
agent = MonteCarloAgent(action_space=env.action_space,
                        obs_space=env.observation_space)

# In[68]:


print(agent.actions)

# ### Setup Blackjack Experiment

# In[65]:



for i_episode in range(1):
    observation = env.reset()
    done = False
    t = 0
    episode = []
    while not done:
        #         print(observation)

        # select action using agent
        action = agent._select_action(score=observation[0],
                                      dealer_card=observation[1],
                                      has_usable_ace=bool2int(observation[2]))

        episode.append(env.step(action))

        #         print("action={}".format(action))
        observation, reward, done, info = env.step(action)
        #         print("observation={}".format(action))
        #         print("reward={}".format(action))
        #         print("done={}".format(action))
        #         print("info={}".format(action))
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

        t += 1

    print(episode)

# In[36]:


env.action_space

# In[ ]:
