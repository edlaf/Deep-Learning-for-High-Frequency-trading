import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import backend.RL_agents.CA as CA


class PPO_clip():
    def __init__(self, No_nothing=False):
        self.policy = CA.ActorCriticAgent(No_nothing=False)
        #self.optimizer = optim.Adam(self.policy.parameters(), lr=self.LR)
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPS_CLIP = 0.2  # Clipping range for PPO
        self.LR = 3e-4  # Learning rate
        self.K_EPOCHS = 4  # Number of optimizeation epochs per update
        self.T_HORIZON = 2048  # Number of timesteps per trajectory segment
        self.M_BATCH_SIZE = 64  # Minibatch size for optimization
        self.num_actors = 8  # Number of actors

    # Function to compute Generalized Advantage Estimation (GAE)
    def compute_gae(self, rewards, values, dones, gamma, lam):
        advantages = np.zeros_like(rewards)
        last_adv = 0
        values = np.append(values, 0)
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]  # 0 if done, 1 otherwise
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
        return advantages, advantages + values[:-1]

    def update(self, memory):
        states, actions, old_probs, rewards, dones, values = zip(*memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        old_log_probs = torch.FloatTensor(old_probs).unsqueeze(1)
        values = torch.FloatTensor(values).unsqueeze(1)
        
        # Compute advantage estimates and targets
        advantages, targets = self.compute_gae(rewards, values.squeeze().tolist(), dones, self.GAMMA, self.LAMBDA)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        targets = torch.FloatTensor(targets).unsqueeze(1)
        
        dataset = list(zip(states, actions, old_log_probs, advantages, targets))

        for _ in range(self.K_EPOCHS):
            np.random.shuffle(dataset)

            # Compute mini batch
            for i in range(0, len(dataset), self.M_BATCH_SIZE):

                batch = dataset[i:i + self.M_BATCH_SIZE]
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_targets = zip(*batch)
                
                batch_states = torch.stack(batch_states)
                batch_actions = torch.stack(batch_actions)
                batch_old_log_probs = torch.stack(batch_old_log_probs)
                batch_advantages = torch.stack(batch_advantages)
                batch_targets = torch.stack(batch_targets)


                # Get new action probabilities and value estimates
                _, new_log_probs, new_values, _, pnl = self.policy.select_action_bis(batch_states)
                new_log_probs = new_log_probs.transpose(0, 1)
                new_values = new_values.squeeze()
                
                # Compute the PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs).squeeze()
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.EPS_CLIP, 1 + self.EPS_CLIP) * batch_advantages
                
                # Compute policy loss (actor)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss (critic)
                value_loss = F.mse_loss(new_values, batch_targets.squeeze())

                # Backpropagation: Actor Update
                self.policy.optimizer_actor.zero_grad()
                policy_loss.backward()
                for param in self.policy.critic.parameters():
                    param.grad = None  # Prevent critic update here
                self.policy.optimizer_actor.step()

                # Backpropagation: Critic Update
                self.policy.optimizer_critic.zero_grad()
                value_loss.backward()
                for param in self.policy.actor.parameters():
                    param.grad = None  # Prevent actor update here
                self.policy.optimizer_critic.step()

    def train(self, visu=True, visu_graph=True, nb_episode=1000, window_size=10, frequency_action=2, comparaison=False):
        """
        Entraîne l'agent sur nb_episode épisodes.
        Améliorations :
        - Normalisation des avantages
        - Bonus d'entropie dans la loss de l'acteur
        - Clipping des gradients
        """
        episode_rewards = []
        all_actor_losses = []
        all_critic_losses = []
        total_losses = []  # pour la visualisation de la loss totale
        episode_actions = []  # pour enregistrer les actions par épisode

        if visu:
            print("                                                            --- Policy Preference Optimization ---\n")
            print(f"\n--- TRAINING THE AGENT OVER {nb_episode} EPISODES OF LENGTH {self.policy.nb_of_action} WITH A FREQUENCY OF {frequency_action} ({int(nb_episode*self.policy.nb_of_action/frequency_action)} training data) ---")
            print("\n     ---> TRAINING...\n")
        
        pbar = tqdm(range(nb_episode), desc="Training Actor-Critic Agent")
        for episode in pbar:
            memory = []
            episode_rewards = []

            for _ in range(self.num_actors):
                state = self.policy.env.reset()
                actor_memory = []
                total_reward = 0
                done = False

                for _ in range(self.T_HORIZON):
                    action, action_log_prob, value, entropy = self.policy.select_action(state)
                    next_state, reward, done, _, pnl = self.policy.env.step(action, frequency_action, No_nothing=self.policy.No_nothing)
                    value = self.policy.critic(torch.FloatTensor(state).unsqueeze(0)).item()
                    actor_memory.append((state, action, action_log_prob, reward, done, value))
                    state = next_state
                    total_reward += reward
                    if done:
                        break

                memory.extend(actor_memory)
                episode_rewards.append(total_reward)

            # Optimize surrogate L wrt theta with K epochs and minibatch size M < NT
            self.update(memory)

            pbar.set_postfix({"Total Reward": f"{total_reward:.2f}"})
        pbar.close()

