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

##############################################
# Réseaux Actor et Critic (inchangés)
##############################################

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)  # Distribution de probabilité sur les actions

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Estimation de la valeur de l'état

##############################################
# Agent Actor-Critic avec méthode de test ajustée
##############################################

class ActorCriticAgent:
    def __init__(self, No_nothing=False):
        # Récupération des paramètres de simulation et de l'environnement
        self.intensity_cancel, self.intensity_order, self.intensity_add, self.price_0, self.tick, self.theta, \
        self.nb_of_action, self.liquidy_last_lim, self.size_max, self.lambda_event, self.event_prob, \
        self.initial_ask, self.initial_bid = param.params_qr()
        
        self.simulation = qr_agent.QrWithAgent(
            self.intensity_cancel, self.intensity_order, self.intensity_add,
            self.price_0, self.tick, self.theta, self.nb_of_action, self.liquidy_last_lim,
            self.size_max, self.lambda_event, self.event_prob
        )
        self.agent = qr_agent.TradingAgent()
        self.nb_steps = self.nb_of_action
        self.env = market.MarketEnv(self.simulation, self.agent, self.initial_ask, self.initial_bid, self.nb_steps)
        
        # Paramètres d'apprentissage (state_dim, action_dim, lr et gamma)
        self.state_dim, self.action_dim, self.lr, self.gamma, _, _, _, _, _, _ = param.params_QDRL(No_nothing=No_nothing)
        
        # Choix du device (CPU / GPU / MPS)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.device = 'cpu'
        
        # Initialisation des réseaux Actor et Critic
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # Optimiseurs pour chaque réseau
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.No_nothing = No_nothing

    def select_action(self, state):
        """
        Pour un état donné, renvoie :
         - l'action échantillonnée depuis la distribution de l'actor,
         - le log-probabilité de cette action (pour le calcul de la loss),
         - et la valeur estimée par le critic.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor)
        return action.item(), log_prob, value

    def train(self, nb_episode=1000, visu=True, frequency_action=1, window_size=10):
        """
        Entraîne l'agent sur nb_episode épisodes.
        - frequency_action est passé à l'environnement via env.step.
        - window_size est utilisé pour le calcul d'une moyenne glissante des récompenses.
        """
        episode_rewards = []
        for episode in range(nb_episode):
            state = self.env.reset()
            done = False
            log_probs = []
            values = []
            rewards = []
            
            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _, pnl = self.env.step(action, frequency_action=frequency_action, No_nothing=self.No_nothing)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state
            
            # Calcul des retours discountés
            returns = []
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Conversion des listes en tenseurs
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)
            
            # Calcul de l'avantage (return - valeur estimée)
            advantages = returns - values.squeeze()
            
            # Calcul des pertes Actor et Critic
            actor_loss = - (log_probs * advantages.detach()).sum()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            loss = actor_loss + critic_loss
            
            # Mise à jour des réseaux
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            if visu:
                print(f"Episode {episode+1}/{nb_episode} - Total Reward: {total_reward:.2f}")
        
        # Visualisation simple avec moyenne glissante
        if visu:
            import matplotlib.pyplot as plt
            import numpy as np
            rolling_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.figure(figsize=(10,5))
            plt.plot(np.arange(len(rolling_avg)), rolling_avg, label='Rolling Average Reward')
            plt.title('Evolution des récompenses')
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()
            
        return episode_rewards

    def test(self, nb_event, frequency_action=1):
        """
        Teste l'agent sur une simulation comportant nb_event événements.
        Le paramètre frequency_action indique le nombre d'événements simulés à chaque appel à env.step.
        """
        print(f"\n--- TESTING THE AGENT OVER A SIMULATION OF {nb_event} EVENTS ---\n")
        state = self.env.reset()
        total_reward = 0.0
        done = False
        event_count = 0
        while not done and event_count < nb_event:
            action, _, _ = self.select_action(state)
            next_state, reward, done, _, pnl = self.env.step(action, frequency_action=frequency_action, No_nothing=self.No_nothing)
            total_reward += reward
            state = next_state
            event_count += frequency_action  # Chaque appel à step correspond à frequency_action événements
        print(f"Test completed: Total Reward: {total_reward:.2f}")
