import numpy as np

def params_qr():
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
    tick = 1
    theta = 0.8
    nb_of_action = 100
    liquidy_last_lim = 50
    size_max = [5, 4, 8]
    lambda_event = [10 for i in range(34)] + [100 for i in range(15)] + [1000]
    event_prob = 1/200
    
    initial_ask = [10, 20, 30]
    initial_bid = [10, 20, 30]
    return intensity_cancel,intensity_order,intensity_add, price_0, tick, theta, nb_of_action, liquidy_last_lim, size_max, lambda_event, event_prob, initial_ask, initial_bid

def params_QDRL(No_nothing = False):
    state_dim = 9
    action_dim = 3
    if No_nothing:
        action_dim = 2
    lr = 1e-3
    gamma = 0.99
    epsilon = 0.7
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 32
    replay_capacity = 10000
    target_update = 10
    return state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, replay_capacity, target_update