import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
import backend.QRModel.QR_only as qr


class TradingAgent:
    def __init__(self, transaction_cost_cancel=0, transaction_cost_market=0):
        self.position = 0
        self.order_active = None
        self.cash = 0.0
        self.cash_depart = 0.0
        self.entry_price = None
        self.transaction_cost_cancel = transaction_cost_cancel
        self.transaction_cost_market = transaction_cost_market
    
    def decide_action(self, market_state):
        """
        Stratégie de base (ici aléatoire).
        Si l'agent n'a pas d'actif, actions possibles : "limit_buy", "cancel_buy"
        Si l'agent a un actif, actions possibles : "market_sell", "limit_sell", "cancel_sell"
        """
        actions = ["do_nothing","order_bid", "order_ask"]

        return np.random.choice(actions)

class QrWithAgent(qr.Qr):
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                liquidy_last_lim, size_max, lambda_event, event_prob):
        super().__init__(tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action,
                        liquidy_last_lim, size_max, lambda_event, event_prob)

    def execute_agent_action(self, action, agent):
        next_size = 1
        time_f = 0
    
        if action == "order_ask":
            if self.ask[0] >= next_size:
                self.ask[0] -= next_size
            else:
                if np.random.uniform(0,1) < self.theta:
                    self.price += self.tick
                    self.ask[0] = self.ask[1]
                    self.ask[1] = self.ask[2]
                    self.ask[2] = self.liquidy_last
                    self.bid[2] = self.bid[1]
                    self.bid[1] = self.bid[0]
                    self.bid[0] = 0
                else:
                    self.ask[0] = 0
            agent.position += 1
            agent.cash -= (self.price + self.tick/2)
            agent.order_active_bid = None
        if action == "order_bid":
            if self.bid[0] >= next_size:
                self.bid[0] -= next_size
            else:
                if np.random.uniform(0,1) < self.theta:
                    self.price -= self.tick
                    self.ask[2] = self.ask[1]
                    self.ask[1] = self.ask[0]
                    self.bid[0] = self.bid[1]
                    self.bid[1] = self.bid[2]
                    self.bid[2] = self.liquidy_last
                    self.ask[0] = 0
                else:
                    self.bid[0] = 0
            agent.position -= 1
            agent.cash += (self.price - self.tick/2)
            agent.order_active_ask = None

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
