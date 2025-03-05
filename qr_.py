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

'''
Market Simulation using a model similar to the Queue Reactive model

Possibility to interact with the market at each step and to create an agent for high frequency strategies using reinforcement learning

It is just a model, and was not meant to reproduce every caracteristics of a real market.
'''

class One_step:
    def __init__(self, tab_cancel, tab_order, tab_add):
        self.n_limit = int(len(tab_add) / 2)
        self.intensities_order  = tab_order
        self.intensities_add    = tab_add
        self.intensities_cancel = tab_cancel
    
    def next_step(self, state, bid, ask, next_size_cancel, next_size_order):
        """
        Modèle purement Poissonien dont l'intensité dépend de l'imbalance.
        Retourne le prochain événement possible :
          - time_f   : temps jusqu'au prochain événement
          - side_f   : côté ('A' pour ask, 'B' pour bid)
          - limit_f  : indice de la limite où se produit l'événement
          - action_f : type d'événement ('Order', 'Add' ou 'Cancel')
        """
        times_order  = []
        action_order = ['A', 'B']
        for i in range(len(self.intensities_order)):
            if action_order[i] == 'A' and ask[0] < next_size_order:
                times_order.append(np.infty)
            elif action_order[i] == 'B' and bid[0] < next_size_order:
                times_order.append(np.infty)
            else:
                times_order.append(np.random.exponential(self.intensities_order[i](state)))
        times_order  = np.array(times_order)
        order_idx = np.argmin(times_order)
        action_order_chosen = action_order[order_idx]
        time_order = times_order[order_idx]
        
        times_add  = []
        action_add = ['A' for _ in range(self.n_limit)] + ['B' for _ in range(self.n_limit)]
        for i in range(len(self.intensities_add)):
            times_add.append(np.random.exponential(self.intensities_add[i](state)))
        times_add  = np.array(times_add)
        add_idx = np.argmin(times_add)
        limit_add  = add_idx % self.n_limit
        action_add_chosen = action_add[add_idx]
        time_add = times_add[add_idx]
        
        times_cancel  = []
        action_cancel = ['A' for _ in range(self.n_limit)] + ['B' for _ in range(self.n_limit)]
        for i in range(len(self.intensities_cancel)):
            if i < self.n_limit:
                if i == self.n_limit - 1 or ask[i] < next_size_cancel:
                    times_cancel.append(np.infty)
                else:
                    times_cancel.append(np.random.exponential(self.intensities_cancel[i](state)))
            else:
                index = i - self.n_limit
                if index == self.n_limit - 1 or bid[index] < next_size_cancel:
                    times_cancel.append(np.infty)
                else:
                    times_cancel.append(np.random.exponential(self.intensities_cancel[i](state)))
        times_cancel  = np.array(times_cancel)
        cancel_idx = np.argmin(times_cancel)
        limit_cancel  = cancel_idx % self.n_limit
        action_cancel_chosen = action_cancel[cancel_idx]
        time_cancel = times_cancel[cancel_idx]
        
        times = np.array([time_order, time_add, time_cancel])
        actions = np.array([action_order_chosen, action_add_chosen, action_cancel_chosen])
        event_idx = np.argmin(times)
        time_f = times[event_idx]
        side_f = actions[event_idx]
        if event_idx == 0:
            limit_f = 0
            action_f = 'Order'
        elif event_idx == 1:
            limit_f = limit_add
            action_f = 'Add'
        else:
            limit_f = limit_cancel
            action_f = 'Cancel'
        return time_f, side_f, limit_f, action_f

class Qr:
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                 liquidy_last_lim, size_max, lambda_event, event_prob):
        """
        Paramètres :
          - tab_cancel, tab_order, tab_add : fonctions d'intensité
          - price_0      : prix initial
          - tick         : tick
          - theta        : paramètre de mean-reversion
          - nb_of_action : nombre d'étapes de la simulation
          - liquidy_last_lim : liquidité du dernier niveau
          - size_max     : [size_max_add, size_max_cancel, size_max_order]
          - lambda_event : temps moyen d'un événement d'actualité
          - event_prob   : probabilité d'un événement
        """
        self.n_limit = int(len(tab_add) / 2)
        self.bid = [0 for _ in range(self.n_limit)]
        self.ask = [0 for _ in range(self.n_limit)]
        self.time = 0
        self.df_evolution = pd.DataFrame(columns=['Time', 'Limit', 'Side', 'Action', 'Price', 'Size',
                                                    'Bid_1', 'Ask_1', 'Bid_2', 'Ask_2', 'Bid_3', 'Ask_3', 'Obs'])
        self.steps = One_step(tab_cancel, tab_order, tab_add)
        self.price = price_0
        self.tick = tick
        self.nb_of_action = nb_of_action
        self.theta = theta
        self.state = 0
        self.liquidy_last = liquidy_last_lim
        self.size_max_add = size_max[0]
        self.size_max_cancel = size_max[1]
        self.size_max_order = size_max[2]
        self.event_prob = event_prob
        self.lambda_event = lambda_event
        self.length_event = 0
        self.is_event = False
        
    def intiate_market(self, initial_ask, initial_bid):
        """Initialise le marché avec les tailles initiales."""
        self.bid = initial_bid.copy()
        self.ask = initial_ask.copy()
        self.time = 0
        self.df_evolution = self.df_evolution.iloc[0:0]
        self.df_evolution.loc[len(self.df_evolution)] = [self.time, 'N/A', 'N/A', 'N/A', self.price, 'N/A',
                                                          self.bid[0], self.ask[0], self.bid[1], self.ask[1],
                                                          self.bid[2], self.ask[2], 'Opening']
    
    def state_(self):
        """
        Calcule l'imbalance en privilégiant le niveau disponible le plus pertinent.
        Par construction, self.ask[2] et self.bid[2] devraient rester égaux à liquidy_last.
        """
        if (self.ask[1] + self.bid[1]) == 0:
            return (self.ask[2] - self.bid[2]) / (self.ask[2] + self.bid[2])
        if (self.ask[0] + self.bid[0]) == 0:
            return (self.ask[1] - self.bid[1]) / (self.ask[1] + self.bid[1])
        return (self.ask[0] - self.bid[0]) / (self.ask[0] + self.bid[0])
    
    def step(self):
        """Simule une étape du marché et met à jour le carnet d'ordres."""
        if not self.is_event:
            if np.random.uniform() > self.event_prob:
                next_size_add = np.random.randint(1, self.size_max_add)
                next_size_cancel = np.random.randint(1, self.size_max_cancel)
                next_size_order = np.random.randint(1, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'N/A'
            else:
                self.is_event = True
                self.length_event = np.random.poisson(self.lambda_event[np.random.randint(len(self.lambda_event))]) + 1
                next_size_add = np.random.randint(1, max(1, self.size_max_add - 2))
                next_size_cancel = np.random.randint(1, max(1, self.size_max_cancel - 2))
                next_size_order = np.random.randint(2, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                time_f = time_f / self.length_event
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'Start_event'
                self.length_event -= 1
        else:
            if self.length_event == 0:
                self.is_event = False
                next_size_add = np.random.randint(1, self.size_max_add)
                next_size_cancel = np.random.randint(1, self.size_max_cancel)
                next_size_order = np.random.randint(1, self.size_max_order)
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'End_event'
            else:
                next_size_add = np.random.randint(1, max(1, self.size_max_add + 1))
                next_size_cancel = np.random.randint(1, max(1, self.size_max_cancel + 1))
                next_size_order = np.random.randint(1, max(1, self.size_max_order - 2))
                time_f, side_f, limit_f, action_f = self.steps.next_step(
                    self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                time_f = time_f / self.length_event
                tab_next_step = [-1 for _ in range(13)]
                tab_next_step[-1] = 'In_event'
                self.length_event -= 1
        
        if side_f == 'A':
            current_ask = self.ask[limit_f]
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_ask = current_ask - next_size_order
                tab_next_step[4] = self.price + self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_ask == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.ask[limit_f] -= next_size_order
            elif action_f == 'Cancel':
                tab_next_step[5] = next_size_cancel
                next_ask = current_ask - next_size_cancel
                tab_next_step[4] = self.price + (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_ask == 0 and limit_f == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.ask[limit_f] -= next_size_cancel
            elif action_f == 'Add':
                tab_next_step[5] = next_size_add
                next_ask = current_ask + next_size_add
                tab_next_step[4] = self.price + (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                self.ask[limit_f] += next_size_add
            
            tab_next_step[-6] = self.ask[0]
            tab_next_step[-7] = self.bid[0]
            tab_next_step[-5] = self.bid[1]
            tab_next_step[-4] = self.ask[1]
            tab_next_step[-2] = self.ask[2]
            tab_next_step[-3] = self.bid[2]
        
        if side_f == 'B':  # Côté Bid
            current_bid = self.bid[limit_f]
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_bid = current_bid - next_size_order
                tab_next_step[4] = self.price - self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_bid == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.bid[limit_f] -= next_size_order
            elif action_f == 'Cancel':
                tab_next_step[5] = next_size_cancel
                next_bid = current_bid - next_size_cancel
                tab_next_step[4] = self.price - (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_bid == 0 and limit_f == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta:
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else:
                        self.price -= self.tick
                        self.ask[2] = self.ask[1]
                        self.ask[1] = self.ask[0]
                        self.ask[0] = 0
                        self.bid[0] = self.bid[1]
                        self.bid[1] = self.bid[2]
                        self.bid[2] = self.liquidy_last
                else:
                    self.bid[limit_f] -= next_size_cancel
            elif action_f == 'Add':
                tab_next_step[5] = next_size_add
                next_bid = current_bid + 1
                tab_next_step[4] = self.price - (1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                self.bid[limit_f] += next_size_add
            
            tab_next_step[-6] = self.ask[0]
            tab_next_step[-7] = self.bid[0]
            tab_next_step[-5] = self.bid[1]
            tab_next_step[-4] = self.ask[1]
            tab_next_step[-2] = self.ask[2]
            tab_next_step[-3] = self.bid[2]
        
        self.time += time_f
        return tab_next_step
    
    def run_market(self, initial_ask, initial_bid):
        """Lance une simulation complète du marché."""
        self.intiate_market(initial_ask, initial_bid)
        for i in range(self.nb_of_action):
            self.df_evolution.loc[len(self.df_evolution)] = self.step()
        return self.df_evolution
    
    def visu(self, initial_ask, initial_bid, price_only):
        """Visualise la simulation du marché."""
        df = self.run_market(initial_ask, initial_bid)
        df_1 = df[df['Limit'] == 0]
        df_price = df_1[df_1['Action'] == 'Order']
        df_2 = df[df['Limit'] == 1]
        df_3 = df[df['Limit'] == 2]
        df_4 = df[df['Obs'].isin(['In_event', 'End_event', 'Start_event'])]
        
        fig = go.Figure()
        if not price_only:
            fig.add_trace(go.Scatter(x=df_1['Time'], y=df_1['Price'], mode='markers', name="Limit_1",
                                     marker=dict(size=5, color="red", opacity=0.7)))
            fig.add_trace(go.Scatter(x=df_2['Time'], y=df_2['Price'], mode='markers', name="Limit_2",
                                     marker=dict(size=4, color="orange", opacity=0.6)))
            fig.add_trace(go.Scatter(x=df_3['Time'], y=df_3['Price'], mode='markers', name="Limit_3",
                                     marker=dict(size=3, color="gold", opacity=0.5)))
            fig.add_trace(go.Scatter(x=df_4['Time'], y=100*np.ones(len(df_4)), mode='markers', name="EVENT",
                                     marker=dict(size=4, color="black", opacity=0.8)))
        fig.add_trace(go.Scatter(x=df_price['Time'], y=df_price['Price'], mode='lines', name="Sell_Price",
                                 line=dict(width=2, color='darkred')))
        fig.update_layout(
            title="Market Simulation",
            xaxis_title="Time",
            yaxis_title="Price",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
    
    def run_and_trade_market(self, initial_ask, initial_bid):
        self.intiate_market(initial_ask, initial_bid)

class TradingAgent:
    def __init__(self, transaction_cost_cancel=0, transaction_cost_market=0):
        self.position = 0
        self.order_active = None
        self.cash = 1000.0
        self.cash_depart = 1000.0
        self.entry_price = None
        self.transaction_cost_cancel = transaction_cost_cancel
        self.transaction_cost_market = transaction_cost_market
    
    def decide_action(self, market_state):
        """
        Stratégie de base (ici aléatoire).
        Si l'agent n'a pas d'actif, actions possibles : "limit_buy", "cancel_buy"
        Si l'agent a un actif, actions possibles : "market_sell", "limit_sell", "cancel_sell"
        """
        if self.position == 0:
            actions = ["do_nothing","limit_buy", "cancel_buy"]
        else:
            actions = ["do_nothing","market_sell", "limit_sell", "cancel_sell"]
        return np.random.choice(actions)

class QrWithAgent(Qr):
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                 liquidy_last_lim, size_max, lambda_event, event_prob):
        super().__init__(tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                         liquidy_last_lim, size_max, lambda_event, event_prob)
    
    def execute_agent_action(self, action, agent):
        if agent.position == 0:
            if action == "limit_buy":
                if agent.order_active is None:
                    self.bid[0] += 1
                    agent.order_active = "buy"
                    # Enregistre le prix d'entrée approximatif
                    if agent.entry_price is None:
                        agent.entry_price = self.price + self.tick/2
                    agent.position = 1
                    # print("Agent placed limit_buy on bid[0] at price:", agent.entry_price)
            elif action == "cancel_buy":
                if agent.order_active == "buy":
                    if self.bid[0] > 0:
                        self.bid[0] -= 1
                    agent.order_active = None
                    # print("Agent canceled limit_buy on bid[0]")
            if self.bid[0] == 0:
                if np.random.uniform(0,1) > self.theta:
                    self.price += self.tick
                    self.ask[0] = self.ask[1]
                    self.ask[1] = self.ask[2]
                    self.ask[2] = self.liquidy_last
                    self.bid[2] = self.bid[1]
                    self.bid[1] = self.bid[0]
                    self.bid[0] = 0
                    # print("Agent triggered new_limit reordering on bid side (price up)")
                else:
                    self.price -= self.tick
                    self.ask[2] = self.ask[1]
                    self.ask[1] = self.ask[0]
                    self.ask[0] = 0
                    self.bid[0] = self.bid[1]
                    self.bid[1] = self.bid[2]
                    self.bid[2] = self.liquidy_last
                    # print("Agent triggered new_limit reordering on bid side (price down)")
        else:
            if action == "market_sell":
                price = self.price - self.tick / 2
                if self.ask[0] >= 1:
                    self.ask[0] -= 1
                    agent.position = 0
                    agent.cash += (price - agent.transaction_cost_market)
                    # print(f"Agent executed market_sell at {price:.2f}")
                    agent.entry_price = None
            elif action == "limit_sell":
                if agent.order_active is None:
                    self.ask[0] += 1
                    agent.order_active = "sell"
                    # print("Agent placed limit_sell on ask[0]")
            elif action == "cancel_sell":
                if agent.order_active == "sell":
                    if self.ask[0] > 0:
                        self.ask[0] -= 1
                    agent.order_active = None
                    agent.cash -= agent.transaction_cost_cancel
                    # print("Agent canceled limit_sell on ask[0]")
            if self.ask[0] == 0:
                if np.random.uniform(0,1) < self.theta:
                    self.price += self.tick
                    self.ask[0] = self.ask[1]
                    self.ask[1] = self.ask[2]
                    self.ask[2] = self.liquidy_last
                    self.bid[2] = self.bid[1]
                    self.bid[1] = self.bid[0]
                    self.bid[0] = 0
                    # print("Agent triggered new_limit reordering on ask side (price up)")
                else:
                    self.price -= self.tick
                    self.ask[2] = self.ask[1]
                    self.ask[1] = self.ask[0]
                    self.ask[0] = 0
                    self.bid[0] = self.bid[1]
                    self.bid[1] = self.bid[2]
                    self.bid[2] = self.liquidy_last
                    # print("Agent triggered new_limit reordering on ask side (price down)")

    def run_market_with_agent(self, initial_ask, initial_bid, agent):
        self.intiate_market(initial_ask, initial_bid)
        for i in range(self.nb_of_action):
            market_step = self.step()
            market_state = {
                "price": self.price,
                "bid": self.bid.copy(),
                "ask": self.ask.copy(),
                "time": self.time
            }
            action = agent.decide_action(market_state)
            print(f"Step {i}: Agent action: {action}")
            self.execute_agent_action(action, agent)
            market_step[-1] = f"Market: {market_step[-1]} | Agent: {action}"
            self.df_evolution.loc[len(self.df_evolution)] = market_step
        return self.df_evolution

class MarketEnv:
    def __init__(self, simulation, agent, initial_ask, initial_bid, nb_steps):
        self.simulation = simulation
        self.agent = agent
        self.initial_ask = initial_ask
        self.initial_bid = initial_bid
        self.nb_steps = nb_steps
        self.current_step = 0
    
    def reset(self):
        self.simulation.intiate_market(self.initial_ask, self.initial_bid)
        self.agent.position = 0
        self.agent.order_active = None
        self.agent.cash = 1000.0
        self.agent.entry_price = None
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        state = np.array([
            self.simulation.price,
            self.simulation.time,
            self.simulation.bid[0],
            self.simulation.bid[1],
            self.simulation.bid[2],
            self.simulation.ask[0],
            self.simulation.ask[1],
            self.simulation.ask[2],
            self.agent.position
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        """
        Pour une étape :
        - Sauvegarde de la valeur nette de l'agent
        - Évolution d'une étape du marché
        - Exécution de l'action de l'agent uniquement tous les 20 pas
        - Calcul de la récompense comme précédemment.
        """
        prev_net = self.agent.cash + self.agent.position * self.simulation.price
        prev_position = self.agent.position

        _ = self.simulation.step()
        self.current_step += 1

        if self.current_step % 20 == 0:
            if self.agent.position == 0:
                action_map = {0: "do_nothing", 1: "limit_buy", 2: "cancel_buy", 3: "do_nothing"}
            else:
                action_map = {0: "do_nothing", 1: "market_sell", 2: "limit_sell", 3: "cancel_sell"}
            action_name = action_map.get(action, "do_nothing")
            self.simulation.execute_agent_action(action_name, self.agent)
        else:
            action_name = "do_nothing"

        new_net = self.agent.cash + self.agent.position * self.simulation.price

        if prev_position == 1 and self.agent.position == 0 and self.agent.entry_price is not None:
            reward = self.simulation.price - self.agent.entry_price
            self.agent.entry_price = None
        else:
            reward = new_net - prev_net

        state = self.get_state()
        done = self.current_step >= self.nb_steps
        return state, reward, done, {}


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

def main():
    def order_bid(x):
        return 0.18*(x**2+1)
    def order_ask(x):
        return 0.18*(x**2+1)
    def tabs_order():
        return [order_ask, order_bid]
    def cancel_bid_1(x):
        return 0.3*(x**2+1)
    def cancel_ask_1(x):
        return 0.3*(x**2+1)
    def cancel_bid_2(x):
        return 0.7*(x**4+1)
    def cancel_ask_2(x):
        return 0.7*(x**4+1)
    def cancel_bid_3(x):
        return 1*(x**4+1)
    def cancel_ask_3(x):
        return 1*(x**4+1)
    def tabs_cancel():
        return [cancel_ask_1, cancel_ask_2, cancel_ask_3, cancel_bid_1, cancel_bid_2, cancel_bid_3]
    def add_bid_1(x):
        return 0.2*(x**4+1)
    def add_ask_1(x):
        return 0.2*(x**4+1)
    def add_bid_2(x):
        return 0.2*(x**2+1)
    def add_ask_2(x):
        return 0.3*(x**2+1)
    def add_bid_3(x):
        return 0.7*(x**2+1)
    def add_ask_3(x):
        return 0.7*(x**2+1)
    def tabs_add():
        return [add_ask_1, add_ask_2, add_ask_3, add_bid_1, add_bid_2, add_bid_3]
    
    tab_add = tabs_add()
    tab_cancel = tabs_cancel()
    tab_order = tabs_order()
    intensity_cancel = tab_cancel
    intensity_order  = tab_order
    intensity_add    = tab_add
    
    price_0 = 100.0
    tick = 0.5
    theta = 0.5
    nb_of_action = 100
    liquidy_last_lim = 50
    size_max = [5, 4, 8]
    lambda_event = [10 for i in range(34)] + [100 for i in range(15)] + [1000]
    event_prob = 1/200
    
    initial_ask = [10, 20, 30]
    initial_bid = [10, 20, 30]
    
    simulation = QrWithAgent(intensity_cancel, intensity_order, intensity_add,
                             price_0, tick, theta, nb_of_action, liquidy_last_lim,
                             size_max, lambda_event, event_prob)
    agent = TradingAgent()
    
    nb_steps = nb_of_action
    env = MarketEnv(simulation, agent, initial_ask, initial_bid, nb_steps)
    
    state_dim = 9
    action_dim = 4
    lr = 1e-3
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    num_episodes = 15000
    batch_size = 32
    replay_capacity = 10000
    target_update = 10
    
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
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
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
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
        if (episode+1) % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
    random_final_rewards = []
    for _ in range(100):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            random_action = random.randrange(action_dim)
            state, reward, done, _ = env.step(random_action)
            total_reward += reward
        random_final_rewards.append(total_reward)
    avg_random_price = np.mean(random_final_rewards)
    print("reward final moyen (stratégie aléatoire) :", avg_random_price)
    plt.plot(episode_rewards)
    plt.plot(np.arange(0,num_episodes,1), np.ones(num_episodes)*avg_random_price)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Évolution des récompenses par épisode")
    plt.show()


    window_size = 20

    rolling_avg = np.convolve(
        episode_rewards, 
        np.ones(window_size) / window_size, 
        mode='valid'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0,num_episodes,1), np.ones(num_episodes)*avg_random_price)
    plt.plot(
        np.arange(window_size - 1, len(episode_rewards)), 
        rolling_avg, 
        color='red', 
        label=f"Rolling Average (window={window_size})"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Évolution des récompenses par épisode (avec moyenne glissante)")
    plt.legend()
    plt.show()



