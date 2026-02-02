import numpy as np
import cv2
import collections
import keras
from keras import layers, models, Sequential, optimizers, losses
import random as rand

class Agent:
    """A dummy class for now that grabs your keyboard input to select actions."""
    def __init__(self, gamma = 0.9):
        self.replay_memory = collections.deque(maxlen=10000)
        self.batch_size = 32
        self.model = Sequential([
            layers.Conv2D(256, kernel_size = (3, 3), strides = 2, padding = 'same', input_shape = (800, 850, 3)),
            layers.ReLU(),
            layers.Conv2D(128, kernel_size = (3, 3), strides = 2, padding = 'same'),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size = (3, 3), strides = 2, padding = 'same'),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size = (3, 3), strides = 2, padding = 'same'),
            layers.Flatten(),
            layers.Dense(4)
        ])
        self.model.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy)

        self.gamma = gamma

    def load_model_weights(self, path):
        self.model.load_weights(path)

    def save_model_weights(self, path):
        self.model.save_weights(path)

    def select_action(self, state):
        """Select an action based on the current state. For now, always returns 0 (no movement)."""
        # Add batch dimension if needed
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        prediction = self.model.predict(state)
        action = np.argmax(prediction)
        return action

    def store_experience(self, state, action, reward, newstate, done, display=False):
        """Store the experience tuple. Dummy implementation displays the state image."""
        if display:
            cv2.imshow("State Image", state)
            cv2.waitKey(1)
        self.replay_memory.append((state, action, reward, newstate, done))

    def train_model(self):
        """Train the model based on stored experiences. Dummy implementation."""
        try:
            sample = rand.sample(self.replay_memory, self.batch_size)
        except ValueError:
            return

        for entry in sample:
            state = entry[0]
            action = entry[1]
            reward = entry[2]
            newstate = entry[3]
            done = entry[4]
            
            # Add batch dimension if needed
            if len(state.shape) == 3:
                state = np.expand_dims(state, axis=0)
            if len(newstate.shape) == 3:
                newstate = np.expand_dims(newstate, axis=0)
            
            predictions_for_sample = self.model.predict(state, verbose=0)
            predictions_for_sample[0][action] = reward
            if not done:
                predicted_reward_for_next_state = np.max(self.model.predict(newstate, verbose=0))
                predicted_reward_for_next_state *= self.gamma
                predictions_for_sample[0][action] += predicted_reward_for_next_state

            self.model.fit(state, predictions_for_sample, verbose=0)