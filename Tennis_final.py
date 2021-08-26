#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q install ./python')


# In[2]:


from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")


# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# In[5]:


for i in range(5):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        #print(rewards) # (1,2)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# In[6]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim,lim)

class Actor(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=256,fc2_units=128):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_agents = 2
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
    
    
class Critic(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=256,fc2_units=128):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_agents = 2
        self.fc1 = nn.Linear(state_size*self.num_agents,fc1_units)
        self.fc2 = nn.Linear(fc1_units+(action_size*self.num_agents),fc2_units)
        self.fc3 = nn.Linear(fc2_units,1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs,action),dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[7]:


import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128 #128
GAMMA = 0.99
TAU = 7e-2
LR_ACTOR = 1e-3
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0
EPSILON = 5.5
EPSILON_DECAY = 1e-4
EPSILON_FINAL = 0.001
MULTIPLE_LEARN_PER_UPDATE = 5
LEARN_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[8]:


import copy
"""Ornstein-Uhlenbeck process."""
class OUNoise:
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


# In[9]:


from collections import namedtuple,deque

class ReplayBuffer:
    def __init__(self,action_size,buffer_size,batch_size,seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)
        self.num_agents = 2
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory,k=self.batch_size)
        states = [torch.from_numpy(np.vstack([e.state[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = [torch.from_numpy(np.vstack([e.next_state[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    

    def __len__(self):
        return len(self.memory)


# In[10]:


import random
class ddpg_agent():
    def __init__(self,state_size,action_size,num_agents,random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)
        
        self.critic_local = Critic(state_size,action_size,random_seed).to(device)
        self.critic_target = Critic(state_size,action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size,random_seed)
        
        self.step_t = 0
        self.epsilon = EPSILON
        
    def reset(self):
        self.noise.reset()
        
        
    def step(self,buffer,agent_number):
        self.step_t = (self.step_t + 1) % LEARN_EVERY
        
        if len(buffer) > BATCH_SIZE:
            if self.step_t == 0:
                experiences = buffer.sample()
                self.learn(experiences,GAMMA,agent_number)
                
    def act(self,state,add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)
    
    def learn(self,experiences,gamma,agent_number):
        states_list,actions_list,rewards,next_states_list,dones = experiences
        
        next_states_tensor = torch.cat(next_states_list,dim=1).to(device)
        states_tensor = torch.cat(states_list,dim=1).to(device)
        actions_tensor = torch.cat(actions_list,dim=1).to(device)
        
        next_actions = [self.actor_target(states) for states in states_list]
        next_actions_tensor = torch.cat(next_actions,dim=1).to(device)
        
        Q_targets_next = self.critic_target(next_states_tensor,next_actions_tensor)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states_tensor,actions_tensor)
        #critic_loss = F.mse_loss(Q_expected,Q_targets[:,agent_number].reshape(-1,1))  Don't use reshape causes the below error
        #RuntimeError: Tensor: invalid storage offset at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCTensor.c:759
        critic_loss = F.mse_loss(Q_expected,Q_targets[:,agent_number].view(-1,1))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        #Actor
        actions_pred = [self.actor_local(states) for states in states_list]
        actions_pred_tensor = torch.cat(actions_pred,dim=1).to(device)
        
        actor_loss = -self.critic_local(states_tensor,actions_pred_tensor).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_local,self.critic_target,TAU)
        self.soft_update(self.actor_local,self.actor_target,TAU)
        
    def soft_update(self,local_model,target_model,tau): 
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        


# In[11]:


class MADDPG():
    def __init__(self,state_size,action_size,num_agents,random_seed):
        self.agents = [ddpg_agent(state_size,action_size,num_agents,random_seed) for x in range(num_agents)]
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
    def step(self,states,actions,next_states,rewards,dones):
        self.memory.add(states,actions,rewards,next_states,dones)
    
        for index,agent in enumerate(self.agents):
            agent.step(self.memory,index)
            
    def act(self,states,add_noise=True):
        actions = np.zeros([self.num_agents,self.action_size])
        
        for index,agent in enumerate(self.agents):
            actions[index,:] = agent.act(states[index],add_noise)
            
        return actions
    
    def checkpoint(self):
        for index,agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor_final.pth'.format(index))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic_final.pth'.format(index))
            
    def reset(self):
        for agent in self.agents:
            agent.reset()


# In[12]:


maddpg_model = MADDPG(state_size,action_size,num_agents,random_seed=0)


# In[13]:


from collections import deque
from workspace_utils import keep_awake
def train_maddpg(n_episodes=5000,print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    average_scores_list = []
    
    for i_episode in keep_awake(range(n_episodes + 1)):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        maddpg_model.reset()
        
        while True:
            actions = maddpg_model.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            maddpg_model.step(states,actions,next_states,rewards,dones)
            states = next_states
            score += rewards
            
            if any(dones):
                break
                
        score_max = np.max(score)
        scores.append(score_max)
        scores_deque.append(score_max)
        average_score = np.mean(scores_deque)
        average_scores_list.append(average_score)
        
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_deque)), end="")
        
        
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, average_score))
            
            if average_score >= 0.5:
                print('\rEnvironment solved in {} episodes with an Average Score of {:.4f}'.format(i_episode, average_score))
                maddpg_model.checkpoint()
                return scores,average_scores_list
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            
    return scores,average_scores_list


# In[14]:


import matplotlib.pyplot as plt
scores,average_scores_list = train_maddpg()
plt.plot(scores)
plt.show()


# In[16]:


plt.plot(scores)
plt.show()


# In[15]:


#env.close()

