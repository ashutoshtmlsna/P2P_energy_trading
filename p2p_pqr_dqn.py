import numpy as np
import pandas as pd
import random
import de_optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple,deque

import matplotlib.pyplot as plt
import time
import resource
import os

import settings

s = settings.s
b = settings.b
loss = settings.loss
r = settings.r
w = settings.w

price_mat = settings.price_mat
bounds = settings.bounds

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy

kp_s = settings.kp_s
kn_s = settings.kn_s
kp_b = settings.kp_b
kn_b = settings.kn_b

zeta_p_s = settings.zeta_p_s
zeta_n_s = settings.zeta_n_s
zeta_p_b = settings.zeta_p_b
zeta_n_b = settings.zeta_n_b

state_space = settings.state_space
state_size = settings.state_size
action_space = settings.action_space
action_size = settings.action_size
# eps = settings.eps
gamma = settings.gamma
lr = settings.lr

y_min = settings.y_min
y_max = settings.y_max


BUFFER_SIZE = int(1000)  # replay buffer size
BATCH_SIZE = 4        # mini-batch size
GAMMA = 0.85            # discount factor
TAU = 0.3              # 0.01 # for soft update of target parameters
ALPHA = 0.00075  # 0.01            # learning rate
UPDATE = 1             # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class QNetwork(nn.Module):
    """ Function Approximator for Actor Model"""

    def __init__(self, state_size, action_size, seed, h1=132, h2=64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            h1 (int): Number of nodes in first hidden layer
            h2 (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        if seed != None:
            self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    def __init__(self, state_size, action_size, seed, h1, h2,i):
        self.state_size = state_size
        self.action_size = action_size
        self.i = i
        if seed == None:
            self.seed = random.seed()
        else:
            self.seed = random.seed(seed)

        ## Q network ##
        self.q_network_l = QNetwork(state_size,action_size, seed, h1, h2).to(device)
        self.q_network_t = QNetwork(state_size,action_size, seed,h1, h2).to(device)

        self.optimizer = optim.Adam(self.q_network_l.parameters(),lr = ALPHA)

        ##Replay buffer ##
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0

    def step(self, state, action, reward, next_step, done):
        self.memory.add(state, action, reward, next_step, done)

        self.timestep = (self.timestep+1)% UPDATE
        if self.timestep == 0:
          if len(self.memory)>BATCH_SIZE:
            expr = self.memory.sample()
            self.learn(expr,GAMMA)

    def action(self, state, eps = 0):
        ''' Returns actions for a given state'''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_l.eval()
        with torch.no_grad():
          action_val = self.q_network_l(state)
        # print("action_val:", action_val)
        self.q_network_l.train()

        ## Epsilon-greedy action selector
        if random.random() > eps:
          # return np.argmax(action_val.cpu().data.numpy())
          return np.random.choice(np.flatnonzero(action_val.cpu().data.numpy() == action_val.cpu().data.numpy().max()))
        else:
          return random.choice(np.arange(self.action_size))

    def learn(self, expr, gamma):
        state, action, reward, next_state, done = expr
        # print(state, action, reward, next_state, done)
        ## Compute Loss
        # criterion = torch.nn.MSELoss()
        # Huber Loss - MSE when small difference and MAE when large
        # criterion = torch.nn.SmoothL1Loss()
        # criterion = ProspectLoss()
        self.q_network_l.train()
        self.q_network_t.eval()

        ## Prediction (64,4)##
        target = self.q_network_l(state).gather(1,action)

        with torch.no_grad():
          next_label = self.q_network_t(next_state).detach().max(1)[0].unsqueeze(1)

        # standard Q-update
        # Q[i, curr_state, action] += lr * td_error
        update = reward/1000 + (GAMMA * next_label*(1-done))
        print("Target:\n", target)
        print("Update:\n",update)
        # print("next_label:\n",next_label)
        # print(i, update.size())

        # Risk Sensitive Q-update
        # with torch.no_grad():
        #     update = update-target
        # update = torch.where(update>0, kp_s[i] * update ** zeta_p_s[i], -kn_s[i] * (-update) ** zeta_n_s[i])
        # update = torch.where(update > 0, kp_s[i] * (update-target) ** zeta_p_s[i], -kn_s[i] * (target-update) ** zeta_n_s[i])

        #print("New Update:\n",update)

        # loss = criterion(target, update).to(device)
        loss = self.my_loss(update,target,self.i)
        # loss = loss.to(device)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # loss.sum().backward()
        self.optimizer.step()

        ## Update the target network ##
        self.soft_update(self.q_network_l, self.q_network_t,TAU)

    def soft_update(self, local, target, tau):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
          t_param.data.copy_(tau*l_param.data + (1-tau)*t_param.data)

    def my_loss(self, input, target,i,beta=0.0):
        # with torch.no_grad():
        update = input-target
        print('Loss:\n',update)
        # print(kp_s[i], zeta_p_s[i], kn_s[i], zeta_n_s[i])
        mask = (update > beta)
        print(mask)
        loss1 = (mask)*(kp_s[i] * abs(update) ** zeta_p_s[i])
        # loss1 = torch.nan_to_num(loss1, nan=1.0)
        print(loss1)
        loss2 = (~mask)*(-kn_s[i] * abs(update) ** zeta_n_s[i])
        # loss2 = torch.nan_to_num(loss2, nan=0.0)
        print(loss2)
        loss = loss1 + loss2
        # loss = torch.nan_to_num(loss, nan=0.0)
        # loss = torch.where(update>0, kp_s[i] * update ** zeta_p_s[i], -kn_s[i] * (-update) ** zeta_n_s[i])
        print(loss)
        return loss.mean()


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        ''' Buffer to store experiences'''
        self.action_size = action_size
        self.mem = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.expr = namedtuple('Experience', field_names = ['state','action','reward', 'next_state', 'done'])
        if seed == None:
            self.seed = random.seed()
        else:
            self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        new_expr = self.expr(state, action, reward, next_state, done)
        self.mem.append(new_expr)

    def sample(self):
        expr = random.sample(self.mem, k=self.batch_size)

        state = torch.from_numpy(np.vstack([exp.state for exp in expr if exp is not None])).float().to(device)
        action = torch.from_numpy(np.vstack([exp.action for exp in expr if exp is not None])).long().to(device)
        reward = torch.from_numpy(np.vstack([exp.reward for exp in expr if exp is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([exp.next_state for exp in expr if exp is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([exp.done for exp in expr if exp is not None]).astype(np.uint8)).float().to(device)

        return (state, action, reward, next_state, done)

    def __len__(self):
        return len(self.mem)



def vi(i, selling_price, seller_cap_sold, r, rho_0):
    return selling_price*seller_cap_sold

def q_learn_price(i, Q, selling_price, seller_cap_sold, day,eps):
    if day == 0:  # selling_price not in state_space:
        abs_val = np.abs(state_space-selling_price)
        smallest_diff = np.argmin(abs_val)
        curr_state = smallest_diff
    else:
        curr_state = np.argwhere(state_space == selling_price)[0][0]

    if random.uniform(0,1) < eps:
        action = random.sample(range(action_size),1)[0] # np.argwhere(Q[i, curr_state,:] == np.random.choice(Q[i, curr_state,:]))[0][0]

    else:
        action = np.random.choice(np.flatnonzero(Q[i, curr_state,:] == Q[i, curr_state,:].max())) #np.argmax(Q[i, curr_state,:])
    # reward = vi(i, selling_price, seller_cap_sold, r=r[i, day], rho_0=target_price_sell[i])
    reward = 0
    # print(curr_state,state_space[curr_state],action, reward, action_space[action])#, new_state, state_space[new_state])
    # print(state_space[curr_state] + action_space[action])
    if y_max >= state_space[curr_state] + action_space[action] >= y_min:
        new_state = np.argwhere(state_space == round(state_space[curr_state] + action_space[action],2))[0][0]
        reward = vi(i, new_state, seller_cap_sold, r=r[i, day], rho_0=target_price_sell[i])
        td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])
    else:
        if state_space[curr_state] + action_space[action] > y_max:  # state_space[curr_state] == 12 and action_space[action] > 0:
            new_state = curr_state
            td_error = 0
        elif state_space[curr_state] + action_space[action] < y_min:  # state_space[curr_state] == 6 and action_space[action] < 0:
            new_state = curr_state
            td_error = 0
        else:
            new_state = np.argwhere(state_space == round(state_space[curr_state] + action_space[action], 2))[0][0]
            reward = vi(i, new_state, seller_cap_sold, r=r[i, day], rho_0=target_price_sell[i])
            td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])

    # standard Q-update
    # Q[i, curr_state, action] += lr * td_error

    # Risk Sensitive Q-update
    if td_error >= 0:
        Q[i, curr_state, action] += lr * kp_s[i] * td_error ** zeta_p_s[i]
    else:
        Q[i, curr_state, action] -= lr * kn_s[i] * (-td_error) ** zeta_n_s[i]
    print(curr_state, state_space[curr_state], action, reward, new_state, state_space[new_state])
    return state_space[new_state]


def dqn(i, agent, selling_price, seller_cap_sold, day,eps):
    curr_state = selling_price
    reward = vi(i,selling_price, seller_cap_sold,r=r[i,day],rho_0=target_price_sell[i])
    action = agent.action(np.array(curr_state),eps)

    if y_max >= curr_state + action_space[action] >= y_min:
        new_state = round(curr_state + action_space[action],2)
        # td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])
    else:
        if curr_state + action_space[action] > y_max:  # state_space[curr_state] == 12 and action_space[action] > 0:
            new_state = curr_state
            #action = np.argwhere(np.array(action_space) == 0)[0][0]
            # td_error = 0
        elif curr_state + action_space[action] < y_min:  # state_space[curr_state] == 6 and action_space[action] < 0:
            new_state = curr_state
            #action = np.argwhere(np.array(action_space) == 0)[0][0]
            # td_error = 0
        else:
            new_state = round(curr_state + action_space[action], 2)
            # td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])

    agent.step(np.array(curr_state), action, reward, np.array(new_state), False)
    print(curr_state, action, reward, new_state)
    return new_state
