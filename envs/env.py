from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import pandas as pd
import itertools
from gymnasium import Env, spaces, utils
from gymnasium.spaces import Tuple, Box, Discrete
from itertools import product



class Env1x2(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, match_data, winners_data, combo_files):
        super().__init__()
        self.odds = match_data 
        self.prizes = winners_data
        self.combo_files = combo_files

        # Define the action space 
        self.action_space = spaces.Discrete(6)  

        # Define the observation space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14, 3), dtype=np.float32)  
        
        self.seasons = sorted(self.odds['Temporada'].unique()) 
        self.season_index = 0  
        self.current_season = None 
        self.current_index = 0  
        self.current_season = None  
        self.terminated = False 
        self.truncated = False  

        # Method to assign bet types, using standard deviation for now ('std_dev' or 'std_dev_sum').
        self.assignation_method = 'std_dev'  

        # Amount of noise to add to state probabilities.
        self.state_noise = 0.1  

        self.combinations = {} 

        # Read each combination file and convert to numpy array.
        for reduction, file in self.combo_files.items():
            self.combinations[reduction] = pd.read_csv(file, header=None).to_numpy()  

        # Defines the strategy, detailing different combinations of bets (triples, doubles) with associated costs and reductions.
        self.strategy = [
            {'triples': 4, 'doubles': 0, 'cost': 6.75, 'reduction': self.combinations[1]},
            {'triples': 0, 'doubles': 7, 'cost': 12.0, 'reduction': self.combinations[2]},
            {'triples': 3, 'doubles': 3, 'cost': 18.0, 'reduction': self.combinations[3]},
            {'triples': 2, 'doubles': 6, 'cost': 48.0, 'reduction': self.combinations[4]},
            {'triples': 8, 'doubles': 0, 'cost': 60.75, 'reduction': self.combinations[5]},
            {'triples': 0, 'doubles': 11, 'cost': 99.0, 'reduction': self.combinations[6]}
        ]
        
    def reset(self, seed=None, options=None):
        # Resets the environment for a new episode. It selects the next season and prepares the state.
        if self.current_season is not None:
            self.season_index = (self.season_index + 1) % len(self.seasons)  # Move to the next season in the cycle.

        self.current_season = self.seasons[self.season_index]  # Set the current season.
        season_filter = self.odds['Temporada'] == self.current_season 
        filtered_data = self.odds[season_filter].reset_index(drop=True)
        self.terminated = False
        self.truncated = False

        data = self.odds.iloc[self.current_index:self.current_index + 14]  # Get the next 14 matches.
        
        state = data[['1', 'X', '2']].to_numpy(dtype=np.float32)  # Extract the betting odds for each match.
        noise_state = np.array([self.add_noise(row) for row in state]).astype(np.float32)  # Add noise to the state.

        return noise_state, {}
    

    def step(self, action):
        #Advances the environment by one step based on the selected action.
        previous_season = self.current_season  # Store the previous season.
        data = self.odds.iloc[self.current_index:self.current_index + 14] 
        state = data[['1', 'X', '2']].to_numpy(dtype=np.float32) 

        reward, info = self.calculate_reward(action, state)  # Calculate the reward and additional info based on the action taken.

        self.current_index += 14  # Move to the next set of 14 matches.
        if self.current_index >= len(self.odds):
            self.current_index = 0  
            self.current_season = self.seasons[0]  
            self.terminated = True  
            self.truncated = True 

        # Update the season after advancing.
        if self.current_index < len(self.odds) - 14:
            self.current_season = self.odds['Temporada'].iloc[self.current_index]

        # If the season has changed, terminate the episode.
        if self.current_season != previous_season:
            self.terminated = True

        return state, reward, self.terminated, self.truncated, info


    def add_noise(self, probs):
        # Adds noise to the probabilities and normalizes them.
        noise = np.random.uniform(-self.state_noise, self.state_noise, size=probs.shape)
        probs_noise = probs + noise 
        probs_noise = np.clip(probs_noise, 0.0, None) 
        norm_probs = probs_noise / np.sum(probs_noise)

        return norm_probs  


    def calculate_reward(self, action, state):
        # Generates the best combination of signs and calculates the reward based on the number of correct predictions.
        strategy = self.strategy[action]  

        sorted_even_matches = np.argsort(np.std(state, axis=1))  # Sort the matches based on the standard deviation of the odds.

        # Select matches based on the chosen assignation method ('std_dev' or 'std_dev_sum').
        if self.assignation_method == 'std_dev':
            sorted_indices = sorted_even_matches[:strategy['triples']+strategy['doubles']] 
        elif self.assignation_method == 'std_dev_sum':
            triples_indices = sorted_even_matches[:strategy['triples']]

            # Process remaining rows after selecting triples.
            other_indices = sorted_even_matches[strategy['triples']:]

            # Sort the remaining rows by the largest sum of two entries.
            other_rows = state[other_indices]
            sorted_row_values = np.sort(other_rows, axis=1)
            top_two_values = sorted_row_values[:, -2:]
            sums_of_top_two = np.sum(top_two_values, axis=1)

            sorted_other_indices = other_indices[np.argsort(-sums_of_top_two)]

            # Select the best matches based on the sorted sums.
            doubles_indices = sorted_other_indices[:strategy['doubles']]

            # The remaining indices stay the same.
            final_other_indices = sorted_other_indices[strategy['doubles']:]

            # Final sorted indices, combining triples, doubles, and other matches.
            sorted_indices = np.concatenate([triples_indices, doubles_indices, final_other_indices])


        # Get the reduction combinations based on the strategy.
        reduction_data = strategy['reduction']
        num_rows, num_cols = reduction_data.shape 
        
        state_signs = np.array(['1', 'X', '2'])[np.argmax(state, axis=1)]
        result = np.tile(state_signs[:, None], (1, num_cols))
        
        # Replace the rows based on the sorted indices.
        for i, idx in enumerate(sorted_indices):
            if self.assignation_method == 'std_dev':
                if idx in sorted_indices:
                    result[idx] = reduction_data[i]
            if self.assignation_method == 'std_dev_sum':
                if idx in triples_indices:
                    result[idx] = reduction_data[i]
                elif idx in doubles_indices:
                    two_max_idx = np.argsort(state[idx])[-2:]
                    result[idx] = reduction_data[i]
                    if 0 in two_max_idx and 2 in two_max_idx: 
                        # If 'X' is the lowest, replace 'X' with '2'.
                        result[idx] = ['2' if x == 'X' else x for x in result[idx]]
                    elif 1 in two_max_idx and 2 in two_max_idx: 
                        # If '1' is not the largest, swap '1' with 'X' and 'X' with '2'.
                        result[idx] = ['X' if x == '1' else '2' if x == 'X' else x for x in result[idx]]

        # Calculate the number of correct hits and find the best combination.
        max_hits = 0 
        best_comb = None 
        matches = self.odds.iloc[self.current_index:self.current_index + 14]  # Get the actual match results.
        matches = matches.reset_index(drop=True) 

        # Iterate over each column to find the best combination.
        for col_idx in range(result.shape[1]):
            column = result[:, col_idx] 
            hits = sum(column[i] == matches['Resultat'][i] for i in range(len(matches)))  # Count the number of correct hits.
            if hits > max_hits:  # Update if a new best combination is found.
                max_hits = hits
                best_comb = column 

        # Calculate the reward based on the number of correct hits.
        if max_hits >= 10:
            jornada = self.odds.iloc[self.current_index]['Jornada']  
            prizes_row = self.prizes[(self.prizes['Temporada'] == self.current_season) &
                                    (self.prizes['Jornada'] == jornada)]  
            factor = 0.16 if max_hits == 14 else 0.09 if max_hits == 10 else 0.075  
            winners = prizes_row[str(max_hits)].iloc[0] + 1  # Get the number of winners for this matchday.

            recap = prizes_row['Recaptació'].iloc[0]  # Get the total revenue for the matchday.
            reward = (recap * factor) / winners - strategy['cost']  # Calculate the reward 
        else:
            reward = -strategy['cost'] 

        matchday = str(matches['ID'].iloc[0])[:-2]  # Extract the matchday ID.

        info = {
            'matchday': matchday,
            'action': action,
            'doubles': strategy['doubles'],
            'triples': strategy['triples'],
            'cost': strategy['cost'],
            'prize': reward + strategy['cost'],
            'hits': max_hits,
            'combination': best_comb
        }

        return reward, info 
