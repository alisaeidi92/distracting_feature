import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import inspect
import functools
import threading
import random



def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

# define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        
        #state_dim, action_dim and action_lim are dynamic inputs based on the agent ID iterator
        
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
	self.action_dim = action_dim
	self.action_lim = action_lim
        
	# network dim [state dim, 256, 128, 64, action dim]
		
	self.fc1 = nn.Linear(state_dim,256)
	self.fc1.weight.data = fanin_init(self.fc1.weight.data.size()) #initializing random normalized weights for states network [layer1]

	self.fc2 = nn.Linear(256,128)
	self.fc2.weight.data = fanin_init(self.fc2.weight.data.size()) #initializing random normalized weights for actions network [layer1]

	self.fc3 = nn.Linear(128,64)
	self.fc3.weight.data = fanin_init(self.fc3.weight.data.size()) #initializing random normalized weights for actions network [layer1]

	self.fc4 = nn.Linear(64,action_dim)
	self.fc4.weight.data.uniform_(0,1)

    def forward(self, x):
        
        x = F.relu(self.fc1(state))
	x = F.relu(self.fc2(x))
	x = F.relu(self.fc3(x))
	action = F.sigmoid(self.fc4(x))

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
	self.action_dim = action_dim
        self.max_action = args.high_action
        
	self.fcs1 = nn.Linear(state_dim,256)
	self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size()) 
	self.fcs2 = nn.Linear(256,128)
	self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size()) 

	self.fca1 = nn.Linear(action_dim,128)
	self.fca1.weight.data = fanin_init(self.fca1.weight.data.size()) 


	self.fc2 = nn.Linear(256,128)
	self.fc2.weight.data = fanin_init(self.fc2.weight.data.size()) 

	self.fc3 = nn.Linear(128,1)
	self.fc3.weight.data.uniform_(-EPS,EPS)
        

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        
        s1 = F.relu(self.fcs1(state))
	s2 = F.relu(self.fcs2(s1))
	a1 = F.relu(self.fca1(action))
	x = torch.cat((s2,a1),dim=1)

	x = F.relu(self.fc2(x))		#using relu activation for the critic (model for regression)
	x = self.fc3(x)
        
        return x


class Buffer:
    def __init__(self, size):
        self.size = size
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['s2_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, s2):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['s2_%d' % i][idxs] = s2[i]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
    

class MADDPG:
	
    def __init__(self, state_dim, action_dim, action_lim, ram, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.state_dim = state_dim
	self.action_dim = action_dim
	self.action_lim = 1
	self.ram = ram
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(self.state_dim, self.action_dim, self.action_lim, agent_id)
        self.critic_network = Critic(self.state_dim, self.action_dim)

        # build up the target network
        self.actor_target_network = Actor(self.state_dim, self.action_dim, self.action_lim, agent_id)
        self.critic_target_network = Critic(self.state_dim, self.action_dim)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # Load the model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
		
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def get_exploitation_action(self, state,alpha_1):
	"""
	gets the action from target actor added with exploration noise
	:param state: state (Numpy array)
	:return: sampled action (Numpy array)
	"""
	state = Variable(torch.from_numpy(state))
	action = self.target_actor.forward(state).detach()
	action=F.softmax(action/alpha_1,0)
	return action.data.numpy()

    """ with noise """
    #def get_exploration_action(self, state,alpha_1):
        #"""
       #gets the action from actor added with exploration noise
	#:param state: state (Numpy array)
	#:return: sampled action (Numpy array)
	#"""
	#state = Variable(torch.from_numpy(state))
	#action = self.actor.forward(state).detach()
	#new_action = action + torch.from_numpy((self.noise.sample() * self.action_lim).astype(np.float32))
	#new_action = F.softmax(new_action/alpha_1, 0)
	#return new_action.data.numpy()
    

    # update the network
    def optimize(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # You only need your own reward during training
        s1, a1, s2 = [], [], []  # Used to install the various items in each agent's experience
        for agent_id in range(self.args.n_agents):
            s1.append(transitions['s1_%d' % agent_id])
            a1.append(transitions['a1_%d' % agent_id])
            s2.append(transitions['s2_%d' % agent_id])

        # calculate the target Q value function
        a2 = []
        with torch.no_grad():
            # Get the action corresponding to the next state
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    a2.append(self.actor_target_network(s2[agent_id]))
                else:
                    # Because the incoming other_agents is one less than the total number, 
		    # it is possible that an agent in the middle is the current agent and cannot be traversed to select actions
                    a2.append(other_agents[index].policy.actor_target_network(s2[agent_id]))
                    index += 1
            next_val = self.critic_target_network(s2, a2).detach()

            y_expected = (r.unsqueeze(1) + self.args.gamma * next_val).detach()

        # the q loss
        y_predicted = self.critic_network(s1, a1)
        loss_critic = (y_expected - y_predicted).pow(2).mean()

        # the actor loss
        # Reselect the action of the current agent in the joint action, and the actions of other agents remain unchanged
        a1[self.agent_id] = self.actor_network(s1[self.agent_id])
        actor_loss = - self.critic_network(s1, a1).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
	return actor_loss.data.numpy(), loss_critic.data.numpy()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
        
"""	
	
if __name__ == '__main__':

	# env = gym.make('BipedalWalker-v2')
	env = gym.make('Pendulum-v0')

	MAX_EPISODES = 5000				# Number of episodes
	MAX_STEPS = 1000				# Max steps at each episode to update
	MAX_BUFFER = 1000000				# Max memory size
	MAX_TOTAL_REWARD = 300				# Not used
	S_DIM = env.observation_space.shape[0]
	A_DIM = env.action_space.shape[0]
	A_MAX = env.action_space.high[0]

	print (' State Dimensions :- ', S_DIM)
	print (' Action Dimensions :- ', A_DIM)
	print (' Action Max :- ', A_MAX)

	ram = Buffer(MAX_BUFFER)			#reply buffer
	trainer = Trainer(S_DIM, A_DIM, A_MAX, ram)	#trainer based on state, action space and memory

	for _ep in range(MAX_EPISODES):
		observation = env.reset()
		print('EPISODE :- ', _ep)
		for r in range(MAX_STEPS):
			env.render()
			state = np.float32(observation)
			
			
			#Select action at = µ(st|θµ) + Nt according to the current policy and exploration noise
			action = trainer.get_exploration_action(state)		
			
			
			# if _ep%5 == 0:
			# 	# validate every 5th episode
			# 	action = trainer.get_exploitation_action(state)
			# else:
			# 	# get action based on observation, use exploration policy here
			# 	action = trainer.get_exploration_action(state)

			
			# Execute action at and observe reward rt and observe new state st+1
			new_observation, reward, done, info = env.step(action) 


			# # dont update if this is validation
			# if _ep%50 == 0 or _ep>450:
			# 	continue

			
			
			if done: 					# if we're done with all the episodes, there's nothing to do
				new_state = None
			else:						#else, save the current s, a, r, st+1 and create a new state)
				new_state = np.float32(new_observation)
				# push this exp in ram
				ram.add(state, action, reward, new_state) # Store transition (st, at, rt, st+1) in R

			observation = new_observation			#update the observation based on the current action

			# perform optimization
			trainer.optimize()
			if done:
				break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)

		if _ep%100 == 0:
			trainer.save_models(_ep)


	print ('Completed episodes')
"""
