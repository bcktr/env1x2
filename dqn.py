import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):

        # Initialization of the neural network for Deep Q-Learning.
        super(DQNetwork, self).__init__()

        # Input layer to hidden layer 1
        self.fc1 = nn.Linear(input_dim, 128)
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(128, 64)
        # Hidden layer 2 to output layer
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        # Definition of a forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class ReplayBuffer:
    def __init__(self, capacity):
        # Initialization of the buffer. When the maximum capacity is reached, older observations are discarded.
        self.buffer = deque(maxlen=capacity)

    def add(self, step_data):
        # Store the collected step data in the buffer
        self.buffer.append(step_data)

    def sample(self, batch_size):
        # Get a random sample from the buffer and return the data
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        # Return the number of stored items in the buffer.
        return len(self.buffer)
    

def train_DQN(network, target_network, buffer, batch_size, gamma, optimizer, loss_fn):
    """
    Function to train the neural network using samples from the ReplayBuffer.
    Args:
        network (DQNetwork): Main network to predict Q-values.
        target_network (DQNetwork): Network to calculate the target Q-values.
        buffer (ReplayBuffer): Replay buffer containing stored data.
        batch_size (int): Number of samples for training.
        gamma (float): Discount factor for future rewards.
        optimizer (torch.optim.Optimizer): Optimizer for the network.
        loss_fn (torch.nn.Module): Loss function (e.g., MSELoss).
    """
    if len(buffer) < batch_size:
        return
    # Obtain a random sample from the ReplayBuffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Convert the data to tensors for PyTorch processing
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Predict Q-values for the actions taken
    q_values = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate the target Q-values
    with torch.no_grad():
        max_next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Calculate the loss between the predicted Q-values and the target Q-values
    loss = loss_fn(q_values, target_q_values)

    # Optimize the main network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
