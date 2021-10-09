import numpy as np
import torch
import os
from MADDPG import MADDPG


class Agent:
    def __init__(self, agent_id, state_dim, action_dim, action_lim, ram):
        self.agent_id = agent_id
        self.policy = MADDPG(self, state_dim, action_dim, action_lim, ram, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.action_lim, self.action_lim, self.action_dim[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.action_lim * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.action_lim, self.action_lim)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.optimize(transitions, other_agents)
