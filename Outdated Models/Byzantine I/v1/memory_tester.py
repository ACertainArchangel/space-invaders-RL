"""A test file for analysing memory consumption"""

import psutil
import random as rand

#from pympler import asizeof

def get_remaining_ram_in_gb():
    # Get the available memory in bytes
    available_memory = psutil.virtual_memory().available
    # Convert bytes to GB
    return available_memory / (1024 ** 3)

from keras import Sequential, layers, regularizers, optimizers
from collections import deque
import numpy as np

class relu3_Qagent_linearOut_dOut_l2():

    class_hyperparameter_strings = '''alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions, sample_size_for_TDERR'''.split(", ")

    def __init__(self, alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions, sample_size_for_TDERR): #20 Hyperparameters!
        self.alpha = alpha
        self.epsilon = epsilon_init 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.gamma = gamma 
        self.batch_size = batch_size
        self.sample_size_for_TDERR = sample_size_for_TDERR
        self.model = self.create_model(layer1_size=layer1_size, layer2_size=layer2_size, layer3_size=layer3_size, layer4_size=layer4_size, 
                                       dropout1=dropout1, dropout2=dropout2, dropout3=dropout3, reg1=reg1, reg2=reg2, reg3=reg3, learning_rate=learning_rate, input_shape=input_shape, output_size=actions)
        
        self.memory=deque(maxlen=memory)

    @staticmethod
    def create_model(layer1_size, layer2_size, layer3_size, layer4_size, 
                     dropout1, dropout2, dropout3, reg1, reg2, reg3, learning_rate, input_shape: tuple, output_size: int):
        model = Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Dense(layer1_size, activation="relu", kernel_regularizer=regularizers.l2(reg1)))
        model.add(layers.Dropout(dropout1))
        model.add(layers.Dense(layer2_size, activation="relu", kernel_regularizer=regularizers.l2(reg2)))
        model.add(layers.Dropout(dropout2))
        model.add(layers.Dense(layer3_size, activation="relu", kernel_regularizer=regularizers.l2(reg3)))
        model.add(layers.Dropout(dropout3))
        model.add(layers.Dense(layer4_size, activation="relu"))
        model.add(layers.Dense(output_size, activation="linear"))
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error',  # or another loss function depending on your task
                  metrics=['mae'])
        return model

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)

    def remember(self,   state, action, reward, nextstate,   done=False):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state):
        return np.argmax(self.model.predict(state)[0])
    
    def replay(self): #Adding beta to adjust for bias (self.model.optimiser.learning_rate = something with beta or whatever) might be a good idea
        samples = rand.sample(self.memory, self.sample_size_for_TDERR)

        guesses = [self.model.predict(experience[0].reshape(1, 1, 23)) for experience in samples]

        newguesses = [self.model.predict(experience[3].reshape(1, 1, 23)) for experience in samples]

        priorities = [abs(guess[0][0][experience[1]]-(experience[2] + np.max(newguess)*self.gamma )) for experience, guess, newguess in zip(samples, guesses, newguesses)]

        """Essentially:
        priorities = []
        for experience in self.memory:
            state = experience[0]
            action = experience[1]
            reward = experience[2]
            nextstate = experience[3]
            #done = experience[4]

            reward_from_action_guess = self.model.predict(state)[action]

            actual_reward = reward + np.max(self.model.predict(nextstate))*self.gamma #bootstrapping guess

            td_err = abs(reward_from_action_guess - actual_reward)

            priorities.append(td_err)"""
        
        probabilities = np.array(priorities)**self.alpha
        probabilities /= probabilities.sum()

        indicies = np.random.choice(len(samples), size=self.batch_size, p=probabilities)

        selected_samples=[samples[i] for i in indicies]

        selected_newguesses = [newguesses[i] for i in indicies]

        states = np.array([experience[0].reshape(1, 23) for experience in selected_samples]).reshape(self.batch_size,1,23)

        actions = [experience[1] for experience in selected_samples]

        target_values = [(experience[2] + np.max(newguess)) if not experience[4] else experience[2] for experience, newguess in zip(selected_samples, selected_newguesses)]

        """Essentially:
        target_values = []
        for experience in selected:
            if not experience[4]:
                value = experience[2] + np.max(self.model.predict(experience[3]))
            else:
                value = experience[2]
            target_values.append(value)
        """

        predictions = self.model.predict(states)

        targets = np.array([[reward if i==action else pred for i, pred in enumerate(prediction)] for prediction, action, reward in zip(predictions, actions, target_values)])
        """
        Generates a numpy array of target values by replacing specific predictions with corresponding rewards.
        Essentially:
        targets = []
        for prediction, action, reward in zip(predictions, actions, target_values):
            target = []
            for i, pred in enumerate(prediction):
                target.append(reward if i == action else pred)
            targets.append(target)
        targets = np.array(targets)"""

        self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=2)

print(first:=get_remaining_ram_in_gb())

agent = relu3_Qagent_linearOut_dOut_l2(alpha=0.8, epsilon_init=1, epsilon_decay=0.995, epsilon_min=0.1, gamma=0.995, layer1_size=1024, layer2_size=512, layer3_size=256, layer4_size=128, batch_size=10, learning_rate=0.0001, dropout1=0.5, dropout2=0.5, dropout3=0.5, reg1=0.001, reg2=0.001, reg3=0.001, memory=100000, input_shape=(1,20), actions=4, sample_size_for_TDERR=10)
for i in range(100000):
    agent.remember(np.array([[[1.0]*23]]), 2, 10.0, np.array([[[1.1]*23]]), False)
agent.replay()

print("")
print(second:=get_remaining_ram_in_gb())
[print("") for i in range(100)]
print((first-second))
print("")
print(agent.replay())#So garbage collector doesn't delete it 
print(Sequential, layers, regularizers, optimizers, np, psutil, deque, rand, relu3_Qagent_linearOut_dOut_l2)

#print(asizeof(agent))