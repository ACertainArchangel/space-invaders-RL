import sys
class NullWriter:
    def write(self, arg):
        pass
    def flush(self):
        pass
original = sys.stdout
sys.stdout=NullWriter()
from keras import Sequential, layers, regularizers, optimizers
from collections import deque
import numpy as np
import random as rand
sys.stdout = original

#there are hard coded shapes in here! Get rid of them!!!

class relu3_Qagent_linearOut_dOut_l2():
    """An agent that uses a 4 layer neural network with ReLU activations, dropout, and L2 regularization."""

    class_hyperparameter_strings = '''alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions'''.split(", ")

    def __init__(self, alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions, sample_size_for_TDERR, time_step_threshold=None): #20 Hyperparameters!
        self.alpha = alpha
        self.epsilon = epsilon_init 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.gamma = gamma 
        self.batch_size = batch_size
        self.sample_size_for_TDERR = sample_size_for_TDERR
        self.input_shape=input_shape
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
        """Given a file loads weights into the model."""
        self.model.load_weights(path)

    def save(self, path):
        """Saves the model weights to a file."""
        if not path.endswith(".weights.h5"):
            path+=".weights.h5"

        self.model.save_weights(path)

    def epsilon_save(self):
        """Returns the current epsilon value."""
        return self.epsilon

    def remember(self,   state, action, reward, nextstate,   done=False):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state):
        """Returns the action to take given a state using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.model.output_shape[-1])  # Random action
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def replay(self): #Adding beta to adjust for bias (self.model.optimiser.learning_rate = something with beta or whatever) might be a good idea
        """Uses memories to train the model, prioratising based on temporal diference error."""
        if len(self.memory)>=self.sample_size_for_TDERR:
            TDERRSAMPLE = int(self.sample_size_for_TDERR)
        elif self.batch_size<=len(self.memory)<self.sample_size_for_TDERR:
            TDERRSAMPLE = len(self.memory)
        elif len(self.memory)<self.batch_size:
            print("Not enough memories to replay. Skipping.")
            return -1
        elif self.sample_size_for_TDERR<=self.batch_size:
            raise ValueError("YOU SET THE SAMPLE SIZE TOO LOW, DUMMY!!!")
        else:
            raise ValueError("COME DEBUG THIS LINE OF CODE, DUMMY!!!")


        samples = rand.sample(self.memory, TDERRSAMPLE)

        guesses = [self.model.predict(experience[0].reshape(1, 1, 23), verbose=0) for experience in samples]

        newguesses = [self.model.predict(experience[3].reshape(1, 1, 23), verbose=0) for experience in samples]

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

        predictions = self.model.predict(states, verbose=0)

        try:
            targets = np.array([[reward if i==action else pred[i] for i, pred in enumerate(prediction)] for prediction, action, reward in zip(predictions, actions, target_values)])
        except:
            for thing in [[reward if i==action else pred[i] for i, pred in enumerate(prediction)] for prediction, action, reward in zip(predictions, actions, target_values)]:
                print(thing)
            raise Exception("Oh, no... This target array is just not it... COME DEBUG!!!")
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

        try:
            print("Fitting model")
            self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=1)
        except:
            for state, target in zip(states, targets):
                print(state, target)
            raise Exception("There was an erros fitting the model to the targets (printed above).")
        
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


if __name__=="__main__":
    from tuners.tuner import tuner

    tuner = tuner(maximum_capacity=1)#oops this will not leave space for data. Fix that!!!

    tuner.fill_up(gigabytes_reserved=0.5)#Add better verbosity mode!!!

    assert tuner.sessions!=[]

    for session in tuner.sessions:
        for i in range(100000):
            session.agent.remember(np.array([[[1.0]*23]]), 2, 10.0, np.array([[[1.1]*23]]), False)
        session.agent.replay()

        


        
