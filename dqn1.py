import gym
import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random

#hyper parameters

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02

#decay period where epsilon_start will reach epsilon_end over these many steps
EPSILON_DECAY = 10000 
TARGET_UPDATE_FREQ = 1000

class Network(nn.Module):
    
    def __init__(self, env):
        super().__init__()
        
        in_features = int(env.observation_space.shape[0])
        
        self.Net = nn.Sequential(
                    nn.Linear(in_features, 65),
                    nn.Tanh(),
                    nn.Linear(64, env.action_space.n)
        )
        
    def forward(self, x):
        return self.Net(x)
    
    def act(self, obs):
        pass
    

env = gym.make('CartPole-v1')
rew_buffer = deque([0.0], maxlen=100) 
replay_buffer = deque(maxlen=BUFFER_SIZE)

online_net = Network(env)
target_net = Network(env)
optimizer = torch.optim.Adam(online_net.parameters(),lr=5e-4)

target_net.load_state_dict(online_net.state_dict())

state, _ = env.reset()
reward_sum = 0

def select_action(epsilon, Qnet, state):
    action_space = np.arange(env.action_space.n)
    p = np.random.uniform(0,1)
    if (p > epsilon):
        return np.random.choice(action_space)
    else:
        #select greedy action
        q_values = Qnet(state.unsqueeze(0))
        r = torch.argmax(q_values, dim=1)[0]

        return r.detach().item()

for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, done, _,_ =  env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    
    if done:
        state, _ = env.reset()
        
for step in itertools.count():
    epsilon = np.interp(step, [0,EPSILON_DECAY],[EPSILON_START,EPSILON_END])
    
    action = select_action(epsilon, online_net, torch.as_tensor(state, dtype = torch.float32))
    next_state, reward, done, _, _ =  env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    reward_sum += reward
    
    if done:
        state, _ = env.reset()
        rew_buffer.append(reward_sum)
        reward_sum = 0
        
    if len(rew_buffer) > 100:
        if np.mean(rew_buffer) > 190:
            while True:
                action = online_net.act(state)
                state, _, done, _, _ =  env.step(action)
                env.render()
                if done:
                    env.reset()
    
    transitions = random.sample(replay_buffer, BUFFER_SIZE)
    
    states = np.array([i[0] for i in transitions])
    actions = np.array([i[1] for i in transitions])
    rewards = np.array([i[2] for i in transitions])
    new_states = np.array([i[3] for i in transitions])
    dones = np.array([i[4] for i in transitions])
    
    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    new_states_t = torch.as_tensor(new_states, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, dtype = torch.float32)
    
    #compute terget 
    
    target_q = target_net(states_t)
    max_target_q = target_q.max(dim = 1,keepdim = True)[0]
    
    targets = rewards_t.unsqueeze(1) + GAMMA*(1-dones_t.unsqueeze(1))*max_target_q
    
    q_values = online_net(states_t)
    
    action_q_values = torch.gather(q_values, dim = 1, index= actions_t.unsqueeze(1))
    
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
       
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    if step %  TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    
    if step % 1000 == 0:
        print(f"step = {step}")
        print(f"av reward = {np.mean(rew_buffer)}")
        

        

