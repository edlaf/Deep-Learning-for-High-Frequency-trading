import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
    
import backend.QRModel.QR_agent as qr_agent
import backend.QRModel.QR_only as qr
import backend.Market_env.Market as market
import backend.utils.intensity_fct_params as param

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def train(visu=True, nb_episode = 1000, window_size = 10):
    intensity_cancel,intensity_order,intensity_add, price_0, tick, theta, nb_of_action, liquidy_last_lim, size_max, lambda_event, event_prob, initial_ask, initial_bid = param.params_qr()
    
    simulation = qr_agent.QrWithAgent(intensity_cancel, intensity_order, intensity_add,
                            price_0, tick, theta, nb_of_action, liquidy_last_lim,
                            size_max, lambda_event, event_prob)
    agent = qr_agent.TradingAgent()
    
    nb_steps = nb_of_action
    env = market.MarketEnv(simulation, agent, initial_ask, initial_bid, nb_steps)
    

    num_episodes = nb_episode

    state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, replay_capacity, target_update = param.params_QDRL()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)
    
    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randrange(action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()
    
    episode_rewards = []
    print("--- Q-DEEP-REINFORCED AGENT ---\n")
    print("TRAINING...")
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0.0
        done = False
        tab_action = []
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            tab_action.append(action)
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = q_network(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
                target = rewards + gamma * next_q_values * (1 - dones)
                
                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        # print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
        if (episode+1) % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
    print("---TRAINING FINISHED---\n")
    if visu:
        random_final_rewards = []
        
        for _ in range(1000):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                random_action = random.randrange(action_dim)
                state, reward, done, _ = env.step(random_action)
                total_reward += reward
            random_final_rewards.append(total_reward)
        avg_random_price = np.mean(random_final_rewards)
        print("--- VISUALISING REWARD EVOLUTION ---\n")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=episode_rewards, mode='lines', name="Agent", line=dict(width = 2, color = 'darkblue')))
        fig.add_trace(go.Scatter(x=np.arange(0,num_episodes,1), y=np.ones(num_episodes)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 2, color = 'darkred')))
        fig.update_layout(
            title="P&L Agent vs Random",
            xaxis_title="Episodes",
            yaxis_title="P&L",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        
        window_size = window_size
        rolling_avg = np.convolve(
            episode_rewards, 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(window_size - 1, len(episode_rewards)), y=rolling_avg, mode='lines', name="Agent", line=dict(width = 2, color = 'darkblue')))
        fig.add_trace(go.Scatter(x=np.arange(0,num_episodes,1), y=np.ones(num_episodes)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 2, color = 'darkred')))
        fig.update_layout(
            title="P&L Agent vs Random (Sliding Window Rolling Average)",
            xaxis_title="Episodes",
            yaxis_title="P&L",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        print("\n--- STATS ---\n")
        print("Average Reward for the Random Strat :", avg_random_price)
        tab_action = np.array(tab_action)
        print("Mean of the actions taken of the last episode by the agent", np.mean(tab_action),"\n")
        print('________________________________________________________________')