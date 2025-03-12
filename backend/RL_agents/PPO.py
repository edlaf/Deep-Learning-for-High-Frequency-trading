import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import CA


class PPO_clip():
    def __init__(self, state_dim, action_dim):
        self.policy = CA.ActorCriticAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPS_CLIP = 0.2  # Clipping range for PPO
        self.LR = 3e-4  # Learning rate
        self.K_EPOCHS = 4  # Number of optimizeation epochs per update
        self.T_HORIZON = 2048  # Number of timesteps per trajectory segment
        self.M_BATCH_SIZE = 64  # Minibatch size for optimization
        self.num_actors = 8  # Number of actors

    # Function to compute Generalized Advantage Estimation (GAE)
    def compute_gae(rewards, values, dones, gamma, lam):
        advantages = np.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]  # 0 if done, 1 otherwise
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
        return advantages, advantages + values[:-1]

    def update(self, memory):
        states, actions, old_probs, rewards, dones, values = zip(*memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        old_probs = torch.FloatTensor(old_probs).unsqueeze(1)
        values = torch.FloatTensor(values).unsqueeze(1)
        
        # Compute advantage estimates and targets
        advantages, targets = self.compute_gae(rewards, values.squeeze().tolist(), dones, self.GAMMA, self.LAMBDA)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        targets = torch.FloatTensor(targets).unsqueeze(1)
        
        dataset = list(zip(states, actions, old_probs, advantages, targets))

        for _ in range(self.K_EPOCHS):
            np.random.shuffle(dataset)

            # Compute mini batch
            for i in range(0, len(dataset), self.M_BATCH_SIZE):

                batch = dataset[i:i + self.M_BATCH_SIZE]
                batch_states, batch_actions, batch_old_probs, batch_advantages, batch_targets = zip(*batch)
                
                batch_states = torch.stack(batch_states)
                batch_actions = torch.stack(batch_actions)
                batch_old_probs = torch.stack(batch_old_probs)
                batch_advantages = torch.stack(batch_advantages)
                batch_targets = torch.stack(batch_targets)


                # Get new action probabilities and value estimates
                new_probs, new_values = self.policy(states)
                new_probs = new_probs.gather(1, actions)
                new_values = new_values.squeeze()
                
                # Compute the PPO loss
                ratio = new_probs / old_probs
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.EPS_CLIP, 1 + self.EPS_CLIP) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, targets.squeeze())
                
                loss = policy_loss + 0.5 * value_loss
                
                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, visu=True, visu_graph = True, nb_episode=1000, window_size=10, frequency_action=2, comparaison=False):
        for k in range(nb_episode):
            memory = []
            episode_rewards = []

            for _ in range(self.num_actors):
                state = self.env.reset()
                actor_memory = []
                total_reward = 0
                done = False

                for _ in range(self.T_HORIZON):
                    action, action_prob = self.policy.select_action(state, self.epsilon)
                    next_state, reward, done, _, pnl = self.env.step(action, frequency_action, No_nothing=self.No_nothing)
                    value = self.policy.critic(torch.FloatTensor(state).unsqueeze(0)).item()
                    actor_memory.append((state, action, action_prob, reward, done, value))
                    state = next_state
                    total_reward += reward
                    if done:
                        break

                memory.extend(actor_memory)
                episode_rewards.append(total_reward)

            # Optimize surrogate L wrt theta with K epochs and minibatch size M < NT
            self.update(memory)

