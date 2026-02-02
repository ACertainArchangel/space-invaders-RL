import numpy as np
import cv2
import collections
import keras
from keras import layers, models, Sequential, optimizers, losses
import random as rand

# Preprocessed image dimensions (downscaled for efficiency)
INPUT_HEIGHT = 84
INPUT_WIDTH = 84
INPUT_CHANNELS = 3

class Agent:
    """DQN Agent for Space Invaders with epsilon-greedy exploration and target network."""
    
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, 
                 learning_rate=0.0001, target_update_freq=1000):
        self.replay_memory = collections.deque(maxlen=50000)
        self.batch_size = 32
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Target network update frequency
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0
        
        # Build main model (for action selection and training)
        self.model = self._build_model()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate), 
            loss=losses.MeanSquaredError()  # MSE not binary crossentropy you stupid idiot (I'm the stupid idiot, you're fine, dear code reader. I would never call you stupid idiot. Unless you were one, but I would have no way to know that while writing this verbose comment so you're fine. Unless you are reading this and are a stupid idiot, in which case, well, you asked for it.)
        )
        
        # Build target model (for stable Q-value targets)
        self.target_model = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        """Build a CNN architecture suitable for the input size."""
        return Sequential([
            # Input: 84x84x3 (downscaled preprocessed image so we don't have to waste 100x compute on REALLY HI DEF RECTANGLES)
            layers.Conv2D(32, kernel_size=(8, 8), strides=4, padding='same', 
                         input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=(4, 4), strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same'),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(512),
            layers.ReLU(),
            layers.Dense(4)  # 4 actions: no move, left, right, shoot
        ])
    
    def update_target_network(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def load_model_weights(self, path):
        """Load weights into both main and target models."""
        self.model.load_weights(path)
        self.update_target_network()

    def save_model_weights(self, path):
        self.model.save_weights(path)

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy."""
        # Add batch dimension if needed because model expects batches and if we don't have it, it'll crash and I will cry and spend 2 hours debugging again so plz don't remove this line D:
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        
        # Epsilon-greedy exploration
        if training and rand.random() < self.epsilon:
            return rand.randint(0, 3)  # Random action
        
        prediction = self.model.predict(state, verbose=0)
        return np.argmax(prediction)
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_experience(self, state, action, reward, newstate, done, display=False):
        """Store the experience tuple in replay memory."""
        if display:
            cv2.imshow("State Image", state)
            cv2.waitKey(1)
        self.replay_memory.append((state, action, reward, newstate, done))

    def train_model(self):
        """Train the model using experience replay with batch updates."""
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample a batch
        batch = rand.sample(self.replay_memory, self.batch_size)
        
        # Prepare batch arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Get Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Get Q-values for next states from TARGET network (stability!)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Compute target Q-values using Bellman equation
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train on the batch
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Increment step counter and update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()