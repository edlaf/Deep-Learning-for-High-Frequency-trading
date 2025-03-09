import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
import backend.QRModel.QR_only as qr
import backend.QRModel.QR_agent as qr_agent
import numpy as np

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
        self.agent.cash = 0
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
    
    def step(self, action, frequency_action):
        prev_net = self.agent.cash + self.agent.position * self.simulation.price
        _ = self.simulation.step()
        self.current_step += 1

        if self.current_step % frequency_action == 0:
            self.current_step += 1
            action_map = {0: "do_nothing", 1: "order_bid", 2:"order_ask"}
            action_name = action_map.get(action)
            self.simulation.execute_agent_action(action_name, self.agent)
        else:
            action_name = "do_nothing"
        new_net = self.agent.cash + self.agent.position * self.simulation.price
        reward = new_net - prev_net
        state = self.get_state()
        done = self.current_step >= self.nb_steps
        return state, reward, done, {}

    def step_trained(self, action, frequency_action, nb_events):
        prev_net = self.agent.cash + self.agent.position * self.simulation.price
        simulated_step = self.simulation.step()
        self.current_step += 1

        if self.current_step % frequency_action == 0:
            self.current_step += 1
            action_map = {0: "do_nothing", 1: "order_bid", 2:"order_ask"}
            action_name = action_map.get(action)
            self.simulation.execute_agent_action(action_name, self.agent)
        else:
            action_name = "do_nothing"
        new_net = self.agent.cash + self.agent.position * self.simulation.price
        reward = new_net - prev_net
        state = self.get_state()
        done = self.current_step >= nb_events
        return state, reward, done, {}, simulated_step