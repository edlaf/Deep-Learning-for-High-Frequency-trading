import numpy as np
import pandas as pd
import plotly.graph_objects as go

'''
Cropyright Edouard Laferté 

edouard.laferte@polytechnique.edu

Market Simulation using a model similar to the Queue Reactive model

Possibility to interact with the market at each step and to create an agent for high frequency strategies using reinforcement learning

It is just a model, and was not meant to reproduce every caracteristics of a real market.
'''

class One_step:
    def __init__(self, tab_cancel, tab_order, tab_add):
        self.n_limit            = int(len(tab_add)/2) # nombre de limites 2 add par limites
        self.intensities_order  = tab_order # intensités de l'order (ask/bid)
        self.intensities_add    = tab_add # ask 1 / ask 2 / ... / bid 1 / bid 2 ...
        self.intensities_cancel = tab_cancel
    
    def next_step(self, state, bid, ask, next_size_cancel, next_size_order): # State = Imbalance
        '''
        Purely Poissonian model, with instensity depending on the imbalance (of the bid and ask)
        return the next possible move (if no liquidity, not taken in account)
        takes ==>
                state    : imbalance, or any other value that the intesity are fuctntion of (need to be the same of bid and ask)
        
        return ==>
                time_f   : time for the next event
                side_f   : side of the next event (Bid or Ask)
                limit_f  : limit where the action occures
                action_f : type of the event (market order, add or cancel)
        '''
        
        # order
        times_order  = []
        action_order = ['A','B']
        for i in range (len(self.intensities_order)):
            if action_order[i] == 'A' and ask[0] < next_size_order:
                times_order.append(np.infty)
            elif action_order[i] == 'B' and bid[0] < next_size_order:
                times_order.append(np.infty)
            else:
                times_order.append(np.random.exponential(self.intensities_order[i](state)))
        
        times_order  = np.array(times_order)
        action_order = action_order[np.argmin(times_order)]
        times_order  = times_order[np.argmin(times_order)]
        
        # add
        times_add  = []
        action_add = ['A' for _  in range (self.n_limit)] + ['B' for _  in range (self.n_limit)]
        for i in range (len(self.intensities_add)):
            times_add.append(np.random.exponential(self.intensities_add[i](state)))
        
        times_add  = np.array(times_add)
        limit_add  = np.argmin(times_add) % self.n_limit
        action_add = action_add[np.argmin(times_add)]
        times_add  = times_add[np.argmin(times_add)]
        
        # cancel
        times_cancel  = []
        action_cancel = ['A' for _  in range (self.n_limit)] + ['B' for _  in range (self.n_limit)]
        
        for i in range (len(self.intensities_cancel)):
            if action_cancel[i] == 'A' and ask[i] < next_size_cancel:
                times_cancel.append(np.infty)
            elif action_cancel[i] == 'B' and bid[i-self.n_limit] < next_size_cancel:
                times_cancel.append(np.infty)
            else:
                times_cancel.append(np.random.exponential(self.intensities_cancel[i](state)))
        
        times_cancel  = np.array(times_cancel)
        limit_cancel  = np.argmin(times_cancel) % self.n_limit
        action_cancel = action_cancel[np.argmin(times_cancel)]
        times_cancel  = times_cancel[np.argmin(times_cancel)]
        
        # next_move
        times   = np.array([times_order, times_add, times_cancel])
        actions = np.array([action_order, action_add, action_cancel])
        argm    = np.argmin(times)
        time_f  = times[argm]
        side_f  = actions[argm]
        
        if argm == 0 : # order
            limit_f  = 0
            action_f = 'Order'
        
        elif argm == 1 : # add
            limit_f  = limit_add
            action_f = 'Add'
            
        elif argm == 2 : # cancel
            limit_f  = limit_cancel
            action_f = 'Cancel'
            
        return time_f, side_f, limit_f, action_f
        

class Qr:
    def __init__(self, tab_cancel, tab_order, tab_add, price_0, tick, theta, nb_of_action, liquidy_last_lim, size_max, lambda_event, event_prob):
        '''
        takes ==>
                tab_cancel   : cancel intensity function
                tab_add      : function add intensity function
                tab_order    : order intensity function
                price_0      : starting price
                tick         : tick
                theta        : theta of the QR model, prices goes up or down with probability theta (encapsulate the mean-reversion)
                nb_of_action : number of actions
                size_max     : tab of size_max for each action (each size is unformly sampled)
                lambda_event : average time of a news event
                event_prob   : probability of a news event

        creates ==>
                n_limit      : number of limits
                bid          : bid of every limit from 1 to n_limit
                ask          : ask of every limit from 1 to n_limit
                evolution    : Simulated order books contains every actions (timestamp, limit, side, action, price, observation)
                price        : price at time t
                time         : cumulated time
                steps        : object class for each steps
                state        : initial state (imbalance)
        '''
        self.n_limit         = int(len(tab_add)/2)
        self.bid             = [0 for _ in range (self.n_limit)]
        self.ask             = [0 for _ in range (self.n_limit)]
        self.time            = 0
        self.df_evolution    = pd.DataFrame(columns = ['Time', 'Limit', 'Side', 'Action', 'Price','Size', 'Bid_1', 'Ask_1', 'Bid_2', 'Ask_2', 'Bid_3', 'Ask_3','Obs'])
        self.steps           = One_step(tab_cancel, tab_order, tab_add)
        self.price           = price_0
        self.tick            = tick
        self.nb_of_action    = nb_of_action
        self.theta           = theta
        self.state           = 0
        self.liquidy_last    = liquidy_last_lim
        self.size_max_add    = size_max[0]
        self.size_max_cancel = size_max[1]
        self.size_max_order  = size_max[2]
        self.event_prob      = event_prob
        self.lambda_event    = lambda_event
        self.length_event    = 0
        self.is_event        = False
        
        
    def intiate_market(self, initial_ask, initial_bid):
        '''
        Create the first state of the market using initial_ask ==> [size at limit 1 of ask, size at limit 2 of ask, ... size at limit 1 of ask] and initial_bid
        '''
        self.bid = initial_bid
        self.ask = initial_ask
        self.df_evolution.loc[len(self.df_evolution)] = [self.time, 'N/A', 'N/A', 'N/A', self.price, 'N/A', self.bid[0], self.ask[0], self.bid[1], self.ask[1], self.bid[2], self.ask[2], 'Opening']
        
    def state_(self):
        '''
        Computes the imbalance
        '''
        if (self.ask[1]+self.bid[1]) == 0:
            return (self.ask[2]-self.bid[2])/(self.ask[2]+self.bid[2])
        if (self.ask[0]+self.bid[0]) == 0:
            return (self.ask[1]-self.bid[1])/(self.ask[1]+self.bid[1])
        return (self.ask[0]-self.bid[0])/(self.ask[0]+self.bid[0])
    
    def step(self):
        '''
        Gives the next step and modify the price if needed
        '''
        if not self.is_event:
            if np.random.uniform() > self.event_prob: # no event
                next_size_add       = np.random.randint(1,self.size_max_add)
                next_size_cancel    = np.random.randint(1,self.size_max_cancel)
                next_size_order     = np.random.randint(1,self.size_max_order)
                
                time_f, side_f, limit_f, action_f = self.steps.next_step(self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                
                tab_next_step       = [-1 for _ in range (13)]
                tab_next_step[-1]    = 'N/A'
            
            else: # event occurring
                self.is_event = True
                self.length_event = np.random.poisson(self.lambda_event[np.random.randint(len(self.lambda_event))]) + 1
                next_size_add       = np.random.randint(1,max(1, self.size_max_add-2)) # more activity but less sizes
                next_size_cancel    = np.random.randint(1,max(1, self.size_max_cancel-2))
                next_size_order     = np.random.randint(2,self.size_max_order) # more trades
                
                time_f, side_f, limit_f, action_f = self.steps.next_step(self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                time_f = time_f/self.length_event
                tab_next_step       = [-1 for _ in range (13)]
                tab_next_step[-1]   = 'Start_event'
                
                self.length_event  -= 1 # we decrease by 1 the length of the event 
                
        else: # If we currently are in an event
            if self.length_event == 0: # no event for the next step
                self.is_event = False
                next_size_add       = np.random.randint(1,self.size_max_add)
                next_size_cancel    = np.random.randint(1,self.size_max_cancel)
                next_size_order     = np.random.randint(1,self.size_max_order)
                
                time_f, side_f, limit_f, action_f = self.steps.next_step(self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                
                tab_next_step       = [-1 for _ in range (13)]
                tab_next_step[-1]   = 'End_event'
            
            else:
                next_size_add       = np.random.randint(1,max(1, self.size_max_add+1)) # more activity with more size
                next_size_cancel    = np.random.randint(1,max(1, self.size_max_cancel+1))
                next_size_order     = np.random.randint(1,max(1, self.size_max_order-2)) # less trades
                
                time_f, side_f, limit_f, action_f = self.steps.next_step(self.state_(), self.bid, self.ask, next_size_cancel, next_size_order)
                
                # times are shorter:
                time_f = time_f / self.length_event
                
                tab_next_step       = [-1 for _ in range (13)]
                tab_next_step[-1]   = 'In_event'
                self.length_event  -= 1
        
        if side_f == 'A': # Ask
            current_ask = self.ask[limit_f]
            # Achat a price+tick/2
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_ask         = current_ask-next_size_order
                tab_next_step[4] = self.price + self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_ask == 0 :
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta :
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else :
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
                next_ask         = current_ask-next_size_cancel
                tab_next_step[4] = self.price+(1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_ask == 0 and limit_f == 0: # premiere limite vide
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) < self.theta :
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else :
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
                next_ask         = current_ask+next_size_add
                tab_next_step[4] = self.price+(1+limit_f)*self.tick / 2
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
            
        if side_f == 'B': # Bid
            current_ask = self.bid[limit_f]
            # price-tick/2
            if action_f == 'Order':
                tab_next_step[5] = next_size_order
                next_ask         = current_ask-next_size_order
                tab_next_step[4] = self.price - self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[1] = limit_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                if next_ask == 0 :
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta :
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else :
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
                next_ask         = current_ask - next_size_cancel
                tab_next_step[4] = self.price-(1+limit_f)*self.tick / 2
                tab_next_step[0] = self.time + time_f
                tab_next_step[2] = side_f
                tab_next_step[3] = action_f
                tab_next_step[1] = limit_f
                if next_ask == 0 and limit_f == 0:
                    tab_next_step[-1] = 'new_limit'
                    if np.random.uniform(0,1) > self.theta :
                        self.price += self.tick
                        self.ask[0] = self.ask[1]
                        self.ask[1] = self.ask[2]
                        self.ask[2] = self.liquidy_last
                        self.bid[2] = self.bid[1]
                        self.bid[1] = self.bid[0]
                        self.bid[0] = 0
                    else :
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
                next_ask         = current_ask+1
                tab_next_step[4] = self.price-(1+limit_f)*self.tick / 2
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
        '''
        Create an entire market simulation
        '''
        self.intiate_market(initial_ask, initial_bid)
        for i in range (self.nb_of_action):
            self.df_evolution.loc[len(self.df_evolution)] = self.step()
        return self.df_evolution
    
    def visu(self, initial_ask, initial_bid, price_only):
        '''
        Let's visulatize a market simulation
        price_only : True if you only want to see the price
        '''
        df       = self.run_market(initial_ask, initial_bid)
        df_1     = df[df['Limit'] == 0]
        df_price = df_1[df_1['Action'] == 'Order']
        df_2     = df[df['Limit'] == 1]
        df_3     = df[df['Limit'] == 2]
        df_4     = df[df['Obs'].isin(['In_event', 'End_event', 'Start_event'])]
        
        fig = go.Figure()
        if not price_only:
            fig.add_trace(go.Scatter(x=df_1['Time'], y=df_1['Price'], mode='markers', name="Limit_1", marker=dict(size=5, color="red", opacity=0.7)))
            fig.add_trace(go.Scatter(x=df_2['Time'], y=df_2['Price'], mode='markers', name="Limit_2", marker=dict(size=4, color="orange", opacity=0.6)))
            fig.add_trace(go.Scatter(x=df_3['Time'], y=df_3['Price'], mode='markers', name="Limit_3", marker=dict(size=3, color="gold", opacity=0.5)))
            fig.add_trace(go.Scatter(x=df_4['Time'], y=100*np.ones(len(df_4)), mode='markers', name="EVENT", marker=dict(size=4,color="black", opacity=0.8)))
        fig.add_trace(go.Scatter(x=df_price['Time'], y=df_price['Price'], mode='lines', name="Sell_Price", line=dict(width = 2, color = 'darkred')))
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
        