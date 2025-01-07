import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import numpy as np
import gymnasium
import gymnasium_env
from dqn import DQNetwork
import matplotlib.pyplot as plt


# Initial configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load test data
match_data = pd.read_csv('data/test_odds.csv', sep=';', decimal=',')
winners_data = pd.read_csv('data/prizes.csv', sep=';', decimal=',')

combo_files = {
    1: 'data/reduction_1.csv',
    2: 'data/reduction_2.csv',
    3: 'data/reduction_3.csv',
    4: 'data/reduction_4.csv',
    5: 'data/reduction_5.csv',
    6: 'data/reduction_6.csv'
}

# Load the test environment with corresponding data
env = gymnasium.make('gymnasium_env/main', match_data=match_data, winners_data=winners_data, combo_files=combo_files)

# Model configuration
input_dim = 14 * 3 # Observation dimension (14 matchdays * 3 features per matchday)
output_dim = env.action_space.n  # Number of possible actions (depending on the environment's action space)

# Load the trained model
network = DQNetwork(input_dim, output_dim)
network.load_state_dict(torch.load('data/model_dqn.pth', weights_only=True))
network.eval()

# Evaluating the model
evaluation_data = []
reward_history = []
total_reward = 0

for episode in range(2):
    state, _ = env.reset()
    state = state.flatten()
    total_steps, reward = 0, 0

    while True:
        # Predict the best action based on the model (greedy strategy)
        with torch.no_grad():
            q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values).item()

        # Execute the action and get the next state and reward
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.flatten()

        total_steps += 1
        state = next_state
        print(
            f"Ep.{episode+1} - St.{total_steps:02d}, Md.{info.get('matchday', 'N/A')} | "
            f"Action: {info.get('action', 'N/A')} ({info.get('doubles', 'N/A'):02d} D {info.get('triples', 'N/A'):02d} T) | "
            f"Hits: {info.get('hits', 'N/A'):02d}, Reward: {reward:.2f}"
        )

        total_reward += reward
        reward_history.append(total_reward)

        if terminated or truncated:
            break

        # Store the information for each step
        evaluation_data.append({
            "Simulació": episode + 1,
            "Passos": total_steps,
            "Jornada": info['matchday'],
            "Dobles": info['doubles'],
            "Triples": info['triples'],
            "Preu": info['cost'],
            "Encerts": info['hits'],
            "Premi": round(info['prize'], 2),
            "Recompensa": round(reward, 2)
        })

    print(
        f"Ep.{episode+1} - St.{total_steps:02d}, Md.{info.get('matchday', 'N/A')} | "
        f"Action: {info.get('action', 'N/A')} ({info.get('doubles', 'N/A'):02d} D {info.get('triples', 'N/A'):02d} T) | "
        f"Hits: {info.get('hits', 'N/A'):02d}, Reward: {reward:.2f}"
    )

# Convert stored data to a DataFrame
df = pd.DataFrame(evaluation_data)
total_reward_df = df["Recompensa"].sum()

print("-----------------------------------------------------------------")
print(f"Temporades: 2 # Recompensa mitjana: {total_reward_df/total_steps:.2f} - Recompensa total: {(total_reward_df):.2f}")
print("-----------------------------------------------------------------")

df.loc[len(df)] = ["Total", "", "", "", "", "", "", "", total_reward_df]  # Add the row at the end
df.to_csv('data/evaluation_results.csv', index=False)

# Convert reward_history to a numpy array for element-wise operations
reward_history = np.array(reward_history)

# Improved graph design
plt.figure(figsize=(12, 6))  # Larger size for better visualization

# Add light gray area under the line
plt.fill_between(range(len(reward_history)), reward_history, color='lightgray', alpha=0.2)

# Draw vertical lines from each point to the origin, with light red or light green depending on the value
for i, reward in enumerate(reward_history):
    color = '#90EE90' if reward >= 0 else '#FFCCCB'  # Light green for positive values, light red for negative
    plt.plot([i, i], [0, reward], color=color, linewidth=2)  # Vertical line from the origin to the point

# Draw the main line thinner
plt.plot(reward_history, color='black', linewidth=1, linestyle='-')  # Black thin line

# Add small dots, no outline, in the appropriate color depending on the value
for i, reward in enumerate(reward_history):
    if reward >= 0:
        plt.scatter(i, reward, color='#228B22', s=5, zorder=3)  # Dark green
    else:
        plt.scatter(i, reward, color='red', s=5, zorder=3)  # Red

# Add a horizontal line at the origin in dark gray
plt.axhline(0, color='darkgray', linewidth=1)

# Axis labels and aesthetics
plt.xlabel("Episodis", fontsize=12)
plt.ylabel("Recompensa", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # Add soft grid
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Final adjustment and visualization
plt.tight_layout()  # Automatically adjust margins to avoid clipping
plt.show()
