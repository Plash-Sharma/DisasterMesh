"""
Q-Learning D2D (Ji et al. Eq. 13, 15)
"""
import numpy as np

class QLearningD2D:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.actions = [0.1, 0.5, 1.0, 2.0, 5.0] # power levels in mW
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return np.random.choice(len(self.actions))
        return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
            
        best_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state][action])
