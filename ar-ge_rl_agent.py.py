# rl_agent.py

import numpy as np
import config

class QLearningAgent:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        # Durum ve aksiyon sayısına göre Q-tablosunu sıfırlarla başlat
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = config.EPSILON

    def choose_action(self, state):
        """Epsilon-greedy stratejisi ile aksiyon seçer."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions) # Keşfet (rastgele seç)
        else:
            return np.argmax(self.q_table[state, :]) # Kullan (en iyi bilineni seç)

    def learn(self, state, action, reward, next_state):
        """Q-tablosunu güncelleme formülünü uygular."""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        # Q-Learning formülü
        new_value = old_value + config.ALPHA * (reward + config.GAMMA * next_max - old_value)
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        """Keşfetme oranını zamanla azaltır."""
        if self.epsilon > config.MIN_EPSILON:
            self.epsilon *= config.EPSILON_DECAY