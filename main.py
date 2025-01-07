import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import gymnasium
import gymnasium_env
import torch
from torch import nn, optim
import numpy as np
from dqn import DQNetwork, ReplayBuffer, train_DQN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter 
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(output_dir='./emissions/')
tracker.start()

try:

    # Seed configuration for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Data loading
    match_data = pd.read_csv('data/odds.csv', sep=';', decimal=',')
    winners_data = pd.read_csv('data/prizes.csv', sep=';', decimal=',')

    # Get unique seasons
    unique_seasons = match_data['Temporada'].unique()
    test_seasons = np.random.choice(unique_seasons, size=2, replace=False)
    train_seasons = [season for season in unique_seasons if season not in test_seasons]
    train_match_data = match_data[match_data['Temporada'].isin(train_seasons)]
    test_match_data = match_data[match_data['Temporada'].isin(test_seasons)]

    # Save train and test datasets for later use
    train_match_data.to_csv('data/train_odds.csv', index=False, sep=';', decimal=',')
    test_match_data.to_csv('data/test_odds.csv', index=False, sep=';', decimal=',')

    combo_files = {
        1: 'data/reduction_1.csv',
        2: 'data/reduction_2.csv',
        3: 'data/reduction_3.csv',
        4: 'data/reduction_4.csv',
        5: 'data/reduction_5.csv',
        6: 'data/reduction_6.csv'
    }

    # Custom environment initialization
    env = gymnasium.make('gymnasium_env/main', match_data=train_match_data, winners_data=winners_data, combo_files=combo_files)

    # Parameter configuration
    num_episodes = 300
    gamma = 0.99  # Discount factor
    batch_size = 64
    learning_rate = 0.001
    target_update_freq = 10
    reward_history = []
    total_reward = 0

    # Epsilon-greedy configuration
    epsilon = 1.0
    epsilon_decay = 0.95
    min_epsilon = 0.05
    epsilon_history = []

    # Neural network (DQN)
    class DQNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(DQNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    # Network initialization
    input_dim = 14 * 3  # The state matrix has dimension 14 x 3
    output_dim = env.action_space.n  # Number of available actions

    network = DQNetwork(input_dim, output_dim)
    target_network = DQNetwork(input_dim, output_dim)
    target_network.load_state_dict(network.state_dict())
    target_network.eval()

    # Optimizer and loss function
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Replay buffer
    buffer = deque(maxlen=10000)

    # Function to store experiences in the buffer
    def store_transition(state, action, reward, next_state, done):
        if next_state is None:
            next_state = np.zeros((14, 3), dtype=np.float32)
        buffer.append((state, action, reward, next_state, done))

    # Function to sample a batch for training
    def sample_batch():
        batch = random.sample(buffer, min(len(buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones


    # Function to train the DQN network
    def train():
        if len(buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = sample_batch()
        q_values = network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
            target = rewards + gamma * next_q_values * (1 - dones)

        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Training loop
    total_reward = 0
    total_reward_hist = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()
        done = False
        episode_reward = 0
        total_steps = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            # Action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = next_state.flatten()
            print(
                f"Ep.{episode+1} - St.{total_steps:02d}, Md.{info.get('matchday', 'N/A')} | "
                f"Action: {info.get('action', 'N/A')} ({info.get('doubles', 'N/A'):02d} D {info.get('triples', 'N/A'):02d} T) | "
                f"Hits: {info.get('hits', 'N/A'):02d}, Reward: {reward:.2f}"
            )
            # Store transition in the buffer
            store_transition(state, action, reward, next_state, done)

            # Train the network
            train()

            # Update the current state
            state = next_state
            episode_reward += reward
            total_steps += 1

        # Update epsilon
        epsilon_history.append(epsilon)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_reward += episode_reward

        # Synchronize the target network every set number of episodes
        if episode % target_update_freq == 0:
            target_network.load_state_dict(network.state_dict())

        # Store the total reward for the episode
        reward_history.append(episode_reward)
        total_reward_hist.append(total_reward)
        print("-----------------------------------------------------------------")
        print(f"Episode {episode+1}/{num_episodes} (steps = {total_steps}, ε = {epsilon:.2f}) - Total Reward: {episode_reward:.2f}")
        print("-----------------------------------------------------------------")
        
    print(num_episodes, " Episodes, Avg Reward: ", round(total_reward/num_episodes, 2), " - TOTAL REWARD: ", round(total_reward, 2))


    # Save the trained model
    torch.save(network.state_dict(), 'data/model_dqn.pth')
    print("Model successfully saved!")


    def plot_reward(reward_history):

        reward_history = np.array(reward_history)
        plt.figure(figsize=(12, 6)) 
        plt.fill_between(range(len(reward_history)), reward_history, color='lightgray', alpha=0.2)

        for i, reward in enumerate(reward_history):
            color = '#90EE90' if reward >= 0 else '#FFCCCB'
            plt.plot([i, i], [0, reward], color=color, linewidth=2) 

        plt.plot(reward_history, color='black', linewidth=1, linestyle='-')

        for i, reward in enumerate(reward_history):
            if reward >= 0:
                plt.scatter(i, reward, color='#228B22', s=5, zorder=3)
            else:
                plt.scatter(i, reward, color='red', s=5, zorder=3)

        plt.axhline(0, color='darkgray', linewidth=1)

        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)  # Soft grid
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.ticklabel_format(style='plain', axis='both') 

        plt.tight_layout() 
        plt.show()



    def plot_epsilon(epsilon_history):

        plt.figure(figsize=(12, 6))
        plt.gca().set_facecolor('white')
        plt.plot(range(len(epsilon_history)), epsilon_history, color='black', linewidth=2)

        plt.scatter(range(len(epsilon_history)), epsilon_history, color='black', s=2, zorder=3)  # Much smaller points

        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Epsilon value", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)  # Soft grid

        plt.tight_layout()
        plt.show()

    plot_reward(reward_history)
    plot_reward(total_reward_hist)
    plot_epsilon(epsilon_history)

finally:
     tracker.stop()
