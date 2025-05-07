from keras import Sequential, layers, regularizers
from collections import deque
import numpy as np


class relu3_Qagent_linearOut_dOut_l2():

    class_hyperparameter_strings = '''alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions'''.split(", ")

    def __init__(self, alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions): #20 Hyperparameters!
        self.alpha = alpha
        self.epsilon = epsilon_init 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.gamma = gamma 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self.create_model(layer1_size, layer2_size, layer3_size, layer4_size, 
                                       dropout1, dropout2, dropout3, reg1, reg2, reg3, input_shape, actions)
        
        self.memory=deque(maxlen=memory)

    @staticmethod
    def create_model(layer1_size, layer2_size, layer3_size, layer4_size, 
                     dropout1, dropout2, dropout3, reg1, reg2, reg3, input_shape: tuple, output_size: int):
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

        priorities = [abs(self.model.predict(experience[0])[experience[1]]-(experience[2] + np.max(self.model.predict(experience[3]))*self.gamma )) for experience in self.memory]
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

        indicies = np.random.choice(len(probabilities), size=self.batch_size, p=probabilities)

        selected=[self.memory[i] for i in indicies]

        states = np.array([experience[0] for experience in selected])

        actions = [experience[1] for experience in selected]

        target_values = [(experience[2] + np.max(self.model.predict(experience[3]))) if not experience[4] else experience[2] for experience in selected]

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



            

        


        
