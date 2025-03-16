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
# Réseaux Actor et Critic
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
# Agent Actor-Critic avec visualisations supplémentaires
##############################################

class ActorCriticAgent:
    def __init__(self, No_nothing=False):
        # Récupération des paramètres de simulation et de l'environnement
        (self.intensity_cancel, self.intensity_order, self.intensity_add,
         self.price_0, self.tick, self.theta, self.nb_of_action,
         self.liquidy_last_lim, self.size_max, self.lambda_event,
         self.event_prob, self.initial_ask, self.initial_bid) = param.params_qr()
        
        self.simulation = qr_agent.QrWithAgent(
            self.intensity_cancel, self.intensity_order, self.intensity_add,
            self.price_0, self.tick, self.theta, self.nb_of_action, self.liquidy_last_lim,
            self.size_max, self.lambda_event, self.event_prob
        )
        self.agent = qr_agent.TradingAgent(self.price_0)
        self.nb_steps = self.nb_of_action
        self.env = market.MarketEnv(self.simulation, self.agent, self.initial_ask, self.initial_bid, self.nb_steps)
        
        # Paramètres d'apprentissage (state_dim, action_dim, lr et gamma)
        self.state_dim, self.action_dim, self.lr, self.gamma, _, _, _, _, _, _, self.beta = param.params_QDRL(No_nothing=No_nothing)
        
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
        self.average_pnl_random = 0  # sera défini lors du test
        self.average_time = 0
        
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
        entropy = dist.entropy()
        value = self.critic(state_tensor)
        return action.item(), log_prob, value, entropy

    def train(self, nb_episode=1000, frequency_action=1, window_size=10, visu=True, visu_graph=True, comparaison=False, max_grad_norm=0.5):
        """
        Entraîne l'agent sur nb_episode épisodes.
          - frequency_action est passé à l'environnement via env.step.
          - window_size est utilisé pour le calcul d'une moyenne glissante des récompenses.
          - Des visualisations interactives avec Plotly sont affichées à la fin de l'entraînement.
        """
        episode_rewards = []

        if visu:
            print("                                                            --- ACTOR-CRITIC AGENT ---\n")
            print(f"\n--- TRAINING THE AGENT OVER {nb_episode} EPISODES OF LENGTH {self.nb_of_action} WITH A FREQUENCY OF {frequency_action} ({int(nb_episode*self.nb_of_action/frequency_action)} training data) ---")
            print("\n     ---> TRAINING...\n")
        
        tab_action_tot = []
        critic_losses, actor_losses = [], []
        pbar = tqdm(range(nb_episode), desc="           Training Actor-Critic Agent")
        for episode in pbar:
            state = self.env.reset()
            done = False
            log_probs = []
            values = []
            tot_reward = 0.0
            rewards = []
            entropies = []  # pour stocker l'entropie de la distribution d'action
            actions = []    # actions effectuées durant cet épisode
            
            # Interaction avec l'environnement pendant l'épisode
            while not done:
                # ATTENTION : modifiez votre méthode select_action pour qu'elle retourne aussi l'entropie.
                action, log_prob, value, entropy = self.select_action(state)
                actions.append(action)
                next_state, reward, done, _, pnl = self.env.step(action, frequency_action=frequency_action, No_nothing=self.No_nothing)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                tot_reward += pnl
                entropies.append(entropy)
                state = next_state
            
            # Stockage des actions de l'épisode
            tab_action_tot.append(actions)
            
            # Calcul des retours discountés
            returns = []
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Conversion des listes en tenseurs
            values_tensor = torch.cat(values)
            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)
            
            # Calcul de l'avantage et normalisation
            advantages = returns - values_tensor.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Calcul des pertes Actor et Critic avec bonus d'entropie
            actor_loss = - (log_probs_tensor * advantages.detach()).sum() + self.beta * entropies_tensor.sum()
            critic_loss = F.mse_loss(values_tensor.squeeze(), returns)
            loss = actor_loss + critic_loss
            
            # Mise à jour des réseaux avec clipping des gradients
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
            pbar.set_postfix({"Total Reward": f"{total_reward:.2f}"})
        pbar.close()
        
        # ----------------------- Visualisations avec Plotly -----------------------
        random_final_rewards = []
        nb_sim = 1000
        print("\n")
        random_pbar = tqdm(range(nb_sim), desc=f"           Training (Random Agent)")
        for _ in random_pbar:
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                random_action = random.randrange(self.action_dim)
                state, reward, done, _, pnl = self.env.step(random_action, frequency_action, No_nothing = self.No_nothing)
                total_reward += pnl
            random_final_rewards.append(total_reward)
            self.average_time += state[1]/nb_sim
        avg_random_price = np.mean(random_final_rewards)
        self.average_pnl_random = avg_random_price
        random_pbar.close()
        
        if comparaison:
            return episode_rewards
        if visu:
            print("\n     ---> TRAINING FINISHED\n")
        if visu_graph:
            print("--- VISUALISING REWARD AND DECISION EVOLUTION ---\n")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0,len(episode_rewards),1), y=episode_rewards, mode='lines', name="Agent", line=dict(width = 1, color = 'darkblue')))
            fig.add_trace(go.Scatter(x=np.arange(0,nb_episode,1), y=np.ones(nb_episode)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 1, color = 'darkred')))
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
            fig.add_trace(go.Scatter(x=np.arange(0,nb_episode,1), y=np.ones(nb_episode)*avg_random_price, mode='lines', name="Random Strategy", line=dict(width = 1, color = 'darkred')))
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
            
            if not self.No_nothing:
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
            
            else:
                order_asK = []
                order_biD = []
                for i in range (len(tab_action_tot)):
                    current_tab = np.array(tab_action_tot[i])
                    order_biD.append(np.count_nonzero(current_tab == 0))
                    order_asK.append(np.count_nonzero(current_tab == 1))
                nb_of_action_agent = order_biD[-1] + order_asK[-1]
                
                fig = go.Figure()
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
            fig.add_trace(go.Scatter(x=np.arange(0,len(critic_losses),1), y=critic_losses, mode='lines', line=dict(width = 1, color = 'darkblue')))
            fig.update_layout(
                title="Critic Loss Evolution",
                xaxis_title="Episodes",
                yaxis_title="Loss",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
            fig.show()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(0,len(actor_losses),1), y=actor_losses, mode='lines', line=dict(width = 1, color = 'darkblue')))
            fig.update_layout(
                title="Actor Loss Evolution",
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
            if not self.No_nothing:
                print(f"         Do Nothing ---> {np.array(nothing)[-1]/nb_of_action_agent}%")
            
            print(f"          Order Bid ---> {np.array(order_biD)[-1]/nb_of_action_agent}%")
            print(f"          Order Ask ---> {np.array(order_asK)[-1]/nb_of_action_agent}%\n")
            print('________________________________________________________________')

    def test(self, nb_event, frequency_action=1):
        """
        Teste l'agent sur une simulation comportant nb_event événements.
        Enregistre et affiche l'évolution du prix et du P&L avec Plotly.
        """
        print(f"\n--- TESTING THE AGENT OVER A SIMULATION OF {nb_event} EVENTS ---\n")
        state = self.env.reset()
        total_reward = 0.0
        done = False
        event_count = 0
        
        price_evolution = []
        time_evolution = []
        pnl_balance = []
        pnl_time = []
        
        # Listes pour enregistrer les actions et leur temps/prix
        agent_action_nothing_time = []
        agent_action_nothing = []
        agent_action_buy_time = []
        agent_action_buy = []
        agent_action_sell_time = []
        agent_action_sell = []
        price_evolution_time = []
        
        #pbar_2 = tqdm(total=nb_event, desc="Testing")
        while not done:
            action, _, _, _ = self.select_action(state)
            next_state, reward, done, _, simulated_step, pnl = self.env.step_trained(action, frequency_action, nb_event, No_nothing = self.No_nothing)
            state = next_state
            total_reward += pnl
            if not self.No_nothing:
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
            else:
                price_evolution.append(next_state[0])
                price_evolution_time.append(next_state[1])
                if action == 0:
                    agent_action_sell.append(next_state[0])
                    agent_action_sell_time.append(next_state[1])
                if action == 1:
                    agent_action_buy.append(next_state[0])
                    agent_action_buy_time.append(next_state[1])
            for j in range (len(simulated_step)):
                price_evolution.append(simulated_step[j][4])
                price_evolution_time.append(simulated_step[j][0])
            pnl_balance.append(total_reward)
            pnl_time.append(next_state[1])

            #pbar.set_postfix(total_reward=f"{total_reward:.2f}")
        #pbar.close()
        # Si la performance aléatoire n'a pas encore été calculée, on le fait ici
        if self.average_pnl_random is None:
            self.average_pnl_random = self.random_action(nb_episode=nb_event)
        if self.average_time is None:
            self.average_time = time_evolution[-1] if time_evolution else 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_evolution_time, y=price_evolution, name = 'Price',mode='lines', line=dict(width = 1, color = 'black')))
        if not self.No_nothing:
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
        fig.add_trace(go.Scatter(x=pnl_time, y=pnl_balance, name = 'P&L',mode='lines', line=dict(width = 2, color = 'darkred')))
        fig.add_trace(go.Scatter(x=pnl_time, y = self.average_pnl_random*np.array(pnl_time)/self.average_time, name = 'Average P&L with a Random Strategy',mode='lines', line=dict(width = 2, color = 'black')))
        fig.update_layout(
                title=f"P&L Evolution (Trained with trajectories of {self.nb_of_action} events)",
                xaxis_title="Time",
                yaxis_title="P&L",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        
    
    def random_action(self, frequency_action=2, nb_episode=1000):
        """
        Exécute une stratégie aléatoire sur 1000 épisodes et retourne le reward moyen.
        """
        random_final_rewards = []
        pbar = tqdm(range(nb_episode), desc="Random Agent")
        for _ in pbar:
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                random_act = random.randrange(self.action_dim)
                state, reward, done, _, pnl = self.env.step(random_act, frequency_action, No_nothing=self.No_nothing)
                total_reward += pnl
            random_final_rewards.append(total_reward)
        avg_random_price = np.mean(random_final_rewards)
        return avg_random_price
