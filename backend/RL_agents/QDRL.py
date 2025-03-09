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

class QNetwork_Linear(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork_Linear, self).__init__()
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

class Deep_Q_Learning_Agent:
    def __init__(self, network_architecture = 'Classic'):
        self.intensity_cancel,self.intensity_order,self.intensity_add, self.price_0, self.tick, self.theta, self.nb_of_action, self.liquidy_last_lim, self.size_max, self.lambda_event, self.event_prob, self.initial_ask, self.initial_bid = param.params_qr()
        
        self.simulation = qr_agent.QrWithAgent(self.intensity_cancel, self.intensity_order, self.intensity_add,
                                self.price_0, self.tick, self.theta, self.nb_of_action, self.liquidy_last_lim,
                                self.size_max, self.lambda_event, self.event_prob)
        self.agent = qr_agent.TradingAgent()
        
        self.nb_steps = self.nb_of_action
        self.env = market.MarketEnv(self.simulation, self.agent, self.initial_ask, self.initial_bid, self.nb_steps)
        self.state_dim, self.action_dim, self.lr, self.gamma, self.epsilon, self.epsilon_decay, self.epsilon_min, self.batch_size, self.replay_capacity, self.target_update = param.params_QDRL()
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
                
        if network_architecture == 'Classic':
            self.q_network = QNetwork_Linear(self.state_dim, self.action_dim).to(self.device)
            self.target_network = QNetwork_Linear(self.state_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        else:
            print('No Network currently implement with the given name')
        
        self.replay_buffer = ReplayBuffer(self.replay_capacity)
        self.arch = network_architecture
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
    def train(self, visu=True, visu_graph = True, nb_episode = 1000, window_size = 10, frequency_action = 2, comparaison = False):
        num_episodes = nb_episode
        episode_rewards = []
        if visu:
            print("                                                            --- Q-DEEP-REINFORCED AGENT ---\n")
            print(f"\n--- TRAINING THE AGENT OVER {nb_episode} EPISODES ---")
            print("\n     ---> TRAINING...\n")
        tab_action_tot = []
        pbar = tqdm(range(num_episodes), desc=f"           Training ({self.arch} Network)")
        losses = []
        for episode in pbar:
            state = self.env.reset()
            total_reward = 0.0
            done = False
            tab_action = []
            while not done:
                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action, frequency_action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                tab_action.append(action)
                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
                    
                    q_values = self.q_network(states).gather(1, actions)
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                    target = rewards + self.gamma * next_q_values * (1 - dones)
                    
                    loss = F.mse_loss(q_values, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            pbar.set_postfix(total_reward=f"{total_reward:.2f}")
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)
            tab_action_tot.append(tab_action)
            losses.append(loss.item())
            # print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
            if (episode+1) % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
        if comparaison:
            return episode_rewards
        if visu:
            print("\n     ---> TRAINING FINISHED\n")
        if visu_graph:
            random_final_rewards = []
            
            for _ in range(1000):
                state = self.env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    random_action = random.randrange(self.action_dim)
                    state, reward, done, _ = self.env.step(random_action, frequency_action)
                    total_reward += reward
                random_final_rewards.append(total_reward)
            avg_random_price = np.mean(random_final_rewards)
            print("--- VISUALISING REWARD AND DECISION EVOLUTION ---\n")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=episode_rewards, mode='lines', name="Agent", line=dict(width = 1, color = 'darkblue')))
            fig.add_trace(go.Scatter(x=np.arange(0,num_episodes,1), y=np.ones(num_episodes)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 1, color = 'darkred')))
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
            fig.add_trace(go.Scatter(x=np.arange(window_size - 1, len(episode_rewards)), y=rolling_avg, mode='lines', name="Agent", line=dict(width = 1, color = 'darkblue')))
            fig.add_trace(go.Scatter(x=np.arange(0,num_episodes,1), y=np.ones(num_episodes)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 1, color = 'darkred')))
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
            
            nothing = []
            order_asK = []
            order_biD = []
            for i in range (len(tab_action_tot)):
                current_tab = np.array(tab_action_tot[i])
                nothing.append(np.count_nonzero(current_tab == 0))
                order_biD.append(np.count_nonzero(current_tab == 1))
                order_asK.append(np.count_nonzero(current_tab == 2))
            nb_of_action_agent = nothing[-1] + order_biD[-1] + order_asK[-1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=np.array(nothing)/nb_of_action_agent, mode='lines', name="Do_Nothing", opacity=0.6, line=dict(width = 1, color = 'darkblue')))
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=np.array(order_biD)/nb_of_action_agent, mode='lines', name="Order_Bid", opacity=0.6, line=dict(width = 1, color = 'darkred')))
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=np.array(order_asK)/nb_of_action_agent, mode='lines', name="Order_Ask", opacity=0.6, line=dict(width = 1, color = 'darkgreen')))
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=1/3*np.ones(len(np.arange(0,len(episode_rewards),1))), mode='lines', opacity=0.8, name="Theorical Values", line=dict(width = 1, color = 'black')))
            
            fig.update_layout(
                title="Evolution of the decision of the Agent",
                xaxis_title="Episodes",
                yaxis_title="Pourcent of the action taken",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
            fig.show()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0,len(losses),1), y=losses, mode='lines', line=dict(width = 1, color = 'darkblue')))
            fig.update_layout(
                title="NN Loss Evolution",
                xaxis_title="Episodes",
                yaxis_title="Loss",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
            fig.show()
            
            print("\n--- STATS ---\n")
            print(f"Action taken by the agent every {frequency_action}")
            print("Average Reward for the Random Strategy :", avg_random_price)
            
            
            
            print("Actions taken of the last episode by the agent:")
            print(f"         Do Nothing ---> {np.array(nothing)[-1]/nb_of_action_agent}%")
            print(f"          Order Bid ---> {np.array(order_biD)[-1]/nb_of_action_agent}%")
            print(f"          Order Ask ---> {np.array(order_asK)[-1]/nb_of_action_agent}%\n")
            print('________________________________________________________________')
            
            
    def test(self, nb_event, frequency_action = 2):
        print(f"\n--- TESTING THE AGENT OVER A SIMULTION OF {nb_event} ---\n")
        pbar = tqdm(range(nb_event), desc="---> Testing")
        state = self.env.reset()
        total_reward = 0.0
        done = False
        price_evolution           = []
        agent_action_buy          = []
        agent_action_sell         = []
        agent_action_nothing      = []
        price_evolution_time      = []
        agent_action_buy_time     = []
        agent_action_sell_time    = []
        agent_action_nothing_time = []
        pnl_balance = []
        pnl_time = []
        while not done:
            action = self.select_action(state, self.epsilon)
            next_state, reward, done, _, simulated_step = self.env.step_trained(action, frequency_action)
            state = next_state
            total_reward += reward
            price_evolution.append(simulated_step[4])
            price_evolution_time.append(next_state[1])
            pnl_balance.append(total_reward)
            pnl_time.append(next_state[1])
            if action != 0:
                price_evolution.append(next_state[0])
                price_evolution_time.append(next_state[1])
            if action == 0:
                agent_action_nothing.append(next_state[0])
                agent_action_nothing_time.append(next_state[1])
            if action == 1:
                agent_action_sell.append(next_state[0])
                agent_action_sell_time.append(next_state[1])
            if action == 2:
                agent_action_buy.append(next_state[0])
                agent_action_buy_time.append(next_state[1])
            pbar.set_postfix(total_reward=f"{total_reward:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_evolution_time, y=price_evolution, name = 'Price',mode='lines', line=dict(width = 1, color = 'red')))
        fig.add_trace(go.Scatter(x=agent_action_nothing_time, y=agent_action_nothing, name = 'Do Nothing',mode='markers'))
        fig.add_trace(go.Scatter(x=agent_action_buy_time, y=agent_action_buy, name = 'Buy',mode='markers'))
        fig.add_trace(go.Scatter(x=agent_action_sell_time, y=agent_action_sell, name = 'Sell',mode='markers'))
        fig.update_layout(
                title="Price Evolution with the Agent Interaction",
                xaxis_title="Time",
                yaxis_title="Loss",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pnl_time, y=pnl_balance, name = 'P&L',mode='lines', line=dict(width = 1, color = 'red')))
        fig.update_layout(
                title="P&L Evolution",
                xaxis_title="Time",
                yaxis_title="P&L",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        
            
    def random_action(self, frequency_action = 2):
        random_final_rewards = []
        for _ in range(1000):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                random_action = random.randrange(self.action_dim)
                state, reward, done, _ = self.env.step(random_action, frequency_action)
                total_reward += reward
                random_final_rewards.append(total_reward)
        avg_random_price = np.mean(random_final_rewards)
        return avg_random_price

def compare_networks(tab_network, nb_episode = 1000, window_size = 20, frequency_action = 2):
    res = []
    for i in range (len(tab_network)):
        agent = Deep_Q_Learning_Agent(tab_network[i])
        res.append(agent.train(visu = False, frequency_action = frequency_action, visu_graph=False, nb_episode = nb_episode, window_size = window_size, comparaison = True))
    agent = Deep_Q_Learning_Agent(tab_network[0])
    res.append(agent.random_action(frequency_action = frequency_action))
    tab_network.append('Random Strategy')
    fig = go.Figure()
    for i in range (len(tab_network)-1):
        fig.add_trace(go.Scatter(x=np.arange(0,len(res[i]),1), y=res[i], name = tab_network[i], mode='lines', line=dict(width = 1)))
    fig.add_trace(go.Scatter(x=np.arange(0,len(res[-2]),1), y=res[-1]*np.ones(len(np.arange(0,len(res[-2]),1))), name = tab_network[-1], mode='lines', line=dict(width = 1)))
    fig.update_layout(
            title="Comparaison between the different architecture",
            xaxis_title="Episodes",
            yaxis_title="P&L",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
    fig.show()