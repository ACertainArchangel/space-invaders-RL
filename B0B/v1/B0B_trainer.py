number_of_train_y_bits = 3000
time_step_threshold = 1000

hyperdict = {
            "alpha": 0.75,  # Prioraty given to surprises
            "epsilon_init": 1.0,           # Initial exploration rates
            "epsilon_decay": 0.999, # Decay rates for exploration
            "epsilon_min": 0.5,           # Minimum exploration rates
            "gamma": 0.996,           # Discount factors for future rewards
            "layer1_size": 1024,         # Sizes of the first hidden layer
            "layer2_size": 512,         # Sizes of the second hidden layer
            "layer3_size": 256,          # Sizes of the third hidden layer
            "layer4_size": 128,           # Sizes of the fourth hidden layer
            "batch_size": 64,      # Batch sizes to try
            "learning_rate": 0.001, # Learning rates for the optimizer
            "dropout1": 0.3,          # Dropout rates for the first layer
            "dropout2": 0.3,          # Dropout rates for the second layer
            "dropout3": 0.3,         # Dropout rates for the third layer
            "reg1": 0.002,              # L2 regularization strengths for layer 1
            "reg2": 0.002,                # L2 regularization strengths for layer 2
            "reg3": 0.002,                # L2 regularization strengths for layer 3
            "memory": 100000,      # Sizes of the replay memory
            "input_shape": (1, 23),         # Input shape for the model
            "actions": 4,                  # Number of possible actions (e.g., in a reinforcement learning task)
            "sample_size_for_TDERR":200,
    }


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
import pygame
import numpy as np
from joblib import Parallel, delayed

sys.stdout = original

#there are hard coded shapes in here! Get rid of them!!!

class relu3_Qagent_linearOut_dOut_l2():

    class_hyperparameter_strings = '''alpha, epsilon_init, epsilon_decay, epsilon_min, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions'''.split(", ")

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
        if not path.endswith(".weights.h5"):
            path+=".weights.h5"

        self.model.load_weights(path)

    def save(self, path):
        if not path.endswith(".weights.h5"):
            path+=".weights.h5"

        self.model.save_weights(path)

    def remember(self,   state, action, reward, nextstate,   done=False):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.model.output_shape[-1])  # Random action
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def replay(self):
        if len(self.memory) >= self.sample_size_for_TDERR:
            TDERRSAMPLE = int(self.sample_size_for_TDERR)
        elif self.batch_size <= len(self.memory) < self.sample_size_for_TDERR:
            TDERRSAMPLE = len(self.memory)
        elif len(self.memory) < self.batch_size:
            print("Not enough memories to replay. Skipping.")
            return -1
        elif self.sample_size_for_TDERR <= self.batch_size:
            raise ValueError("YOU SET THE SAMPLE SIZE TOO LOW, DUMMY!!!")
        else:
            raise ValueError("COME DEBUG THIS LINE OF CODE, DUMMY!!!")

        samples = rand.sample(self.memory, TDERRSAMPLE)

        # Removed guesses and newguesses
        # guesses = [self.model.predict(experience[0].reshape(1, 1, 23), verbose=0) for experience in samples]
        # newguesses = [self.model.predict(experience[3].reshape(1, 1, 23), verbose=0) for experience in samples]

        # Dummy priorities for the sake of keeping the rest of the code intact
        priorities = [1.0] * TDERRSAMPLE  # Placeholder values

        probabilities = np.array(priorities) ** self.alpha
        probabilities /= probabilities.sum()

        indicies = np.random.choice(len(samples), size=self.batch_size, p=probabilities)

        selected_samples = [samples[i] for i in indicies]

        states = np.array([experience[0].reshape(1, 23) for experience in selected_samples]).reshape(self.batch_size, 1, 23)

        actions = [experience[1] for experience in selected_samples]

        # Calculate target values without using guesses
        target_values = [(experience[2]) for experience in selected_samples]  # Simplified target values

        predictions = self.model.predict(states, verbose=0)

        try:
            targets = np.array([[reward if i == action else pred[i] for i, pred in enumerate(prediction)] for prediction, action, reward in zip(predictions, actions, target_values)])
        except:
            for thing in [[reward if i == action else pred[i] for i, pred in enumerate(prediction)] for prediction, action, reward in zip(predictions, actions, target_values)]:
                print(thing)
            raise Exception("Oh, no... This target array is just not it... COME DEBUG!!!")

        try:
            print("Fitting model")
            self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=1)
        except:
            for state, target in zip(states, targets):
                print(state, target)
            raise Exception("There was an error fitting the model to the targets (printed above).")

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


#Fix bullet render

def is_collision(enemyX, enemyY, bulletX, bulletY, coldist):
    distance = np.linalg.norm(np.array([enemyX, enemyY]) - np.array([bulletX, bulletY]))
    return distance < coldist

class environment():

    pygame_initialized = False

    @classmethod
    def _initialize_pygame(cls):
        pygame.init()
        cls.pygame_initialized=True

    def initialize_rendering(self):
        self.rendering = True
        self.font = pygame.font.Font("resources/TNR.ttf", 32)
        self.gofont = pygame.font.Font("resources/TNR.ttf", 100)
        self.scfont = pygame.font.Font("resources/TNR.ttf", 80)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.Background = pygame.image.load("resources/space.jpg")
        pygame.display.set_caption("Space Invaders")
        icon = pygame.image.load("resources/ufo.png")
        pygame.display.set_icon(icon)
        self.PlayerImg = pygame.image.load("resources/hero.png")

        self.player = lambda x, y: self.screen.blit(self.PlayerImg, (x, y))

        self.bulletImg = pygame.image.load("resources/bolt.png")

        self.enemyImg = pygame.image.load("resources/ufo.png")

        self.textX = 10
        self.textY = 10
        def show_score(x,y):
            score = self.font.render("Score: %d" %self.score_value, True, (255, 255, 255))
            self.screen.blit(score, (x, y))

        def show_ammo(x,y):
            ammo = self.font.render("Ammo: %.1f" %self.ammo_value, True, (255, 255, 255))
            self.screen.blit(ammo, (x, y))

        self.show_score=show_score
        self.show_ammo=show_ammo


    def __init__(self, ammo_inc, Player_Speed, Enemy_Speed, starting_ammo, num_enem, ammo_penalty,
    hit_reward, death_penalty, closeness_penalty, closeness_threshold, SCREEN_HEIGHT=800, SCREEN_WIDTH=850):
        
        self.rendering = False
        
        self.ammo_inc = ammo_inc
        self.Player_Speed = Player_Speed
        self.Enemy_Speed = Enemy_Speed
        self.ammo_value = starting_ammo
        self.starting_ammo = starting_ammo
        self.num_enem = num_enem
        self.ammo_penalty = ammo_penalty#
        self.hit_reward = hit_reward#
        self.death_penalty = death_penalty#
        self.closeness_penalty = closeness_penalty#
        self.closeness_threshold = closeness_threshold#
        self.SCREEN_HEIGHT=SCREEN_HEIGHT
        self.SCREEN_WIDTH=SCREEN_WIDTH

        self._initialize_pygame()

        self.reset()

        self.total_reward = 0

    def reset(self):
        self.game_over = False
        self.playerX = 400
        self.playerY = 730
        self.playerXD = 0

        self.bulletX = 400
        self.bulletY = 730
        self.bulletXD = 0
        self.bulletYD = 20
        self.bullet_state = "ready"

        self.ammo_value=self.starting_ammo

        def fire_bullet(x,y):
            self.bullet_state = "fire"
            if self.rendering:
                self.screen.blit(self.bulletImg, (x, y))
            
        self.fire_bullet=fire_bullet

        self.score_value = 0

        self.enemies=[self.enemy(self) for i in range(self.num_enem)]

        self.reward = 0

    class enemy():
        def __init__(self, parent):
            self.enemyX = rand.randint(0,750)
            self.enemyY=rand.randint(50, 150)
            self.enemyXD=parent.Enemy_Speed*(rand.randint(0,1)*2-1)
            self.enemyYD=40
            self.parent = parent

        def show(self):
            self.parent.screen.blit(self.parent.enemyImg, (self.enemyX, self.enemyY))

        def move(self):
            if self.enemyX<0:
                self.enemyXD= self.parent.Enemy_Speed #0.5
                self.enemyY=self.enemyYD+self.enemyY
        
            elif self.enemyX>786:
                self.enemyXD=-self.parent.Enemy_Speed#-0.5
                self.enemyY=self.enemyYD+self.enemyY
            self.enemyX = self.enemyXD + self.enemyX
        
            collision = is_collision(self.enemyX, self.enemyY, self.parent.bulletX, self.parent.bulletY, 27) and self.parent.bullet_state == "fire"
            if not self.parent.game_over:
                self.parent.game_over = is_collision(self.enemyX,self.enemyY,self.parent.playerX,self.parent.playerY, 33)


            if collision:
                self.parent.reward+=self.parent.hit_reward
                self.parent.total_reward+=self.parent.hit_reward
                self.parent.bullet_state="ready"
                self.parent.bulletY=730
                self.parent.score_value += 1
                self.parent.ammo_value+= self.parent.ammo_inc
                self.enemyX = rand.randint(0,750)
                self.enemyY = rand.randint(50, 150)
                if self.enemyX%2 == 0:
                    self.enemyXD = self.parent.Enemy_Speed #0.5
                else:
                    self.enemyXD = -self.parent.Enemy_Speed #0.5

    def move_player(self):
        if self.playerX<0:
            self.playerX=0
        elif self.playerX>786:
            self.playerX=786
        else:
            self.playerX = self.playerXD + self.playerX

    def move_bullet(self):
        if self.bullet_state == "fire":
            self.fire_bullet(self.bulletX, self.bulletY)
            self.bulletY -= self.bulletYD

            if self.bulletY < 0:
                self.bulletY = 730
                self.bullet_state = "ready"

    def phobia(self):
        """If each enemy has gone farther than the threshold as a fraction of the screen punish the agent"""
        for e in self.enemies:
            if e.enemyY>self.SCREEN_HEIGHT*self.closeness_threshold:
                self.reward-=self.closeness_penalty*e.enemyY/self.SCREEN_HEIGHT
                self.total_reward-=self.closeness_penalty*e.enemyY/self.SCREEN_HEIGHT


    def state(self):
        return np.array(
        [self.playerX, self.playerXD, 1 if self.bullet_state == "ready" else 0, self.bulletX if self.bullet_state=="fire" else 0, 
         self.bulletY if self.bullet_state =="fire" else 0] +
        [value for e in self.enemies for value in (e.enemyX, e.enemyY, e.enemyXD)]
        ).reshape(1,1,23)

    def step(self, action):

        """Takes an action and returns (reward, state, newstate, done)"""

        state = self.state()

        if action==0:
            self.playerXD=0
        elif action==1:
            self.playerXD=-self.Player_Speed
        elif action==2:
            self.playerXD=+self.Player_Speed
        elif action==3 and self.bullet_state=="ready":
            self.bulletX=self.playerX
            self.bullet_state="fire"
            self.ammo_value -= 1 
            self.reward-=self.ammo_penalty
            self.total_reward-=self.ammo_penalty

        if self.ammo_value <1 and self.bullet_state == "ready":
            self.game_over=True

        for e in self.enemies:
            e.move()
        self.move_player()
        self.move_bullet()

        newstate = self.state()

        if self.game_over:
            self.reward-=self.death_penalty
            self.total_reward-=self.death_penalty

        self.phobia()

        reward = float(self.reward)
        self.reward = 0

        gameover = self.game_over

        if self.game_over:
            self.reset() #NO!!! BAD!!! FIX THIS!!!

        return (reward, state, newstate, gameover)

    def render(self):
        self.screen.fill((0,0,0))
        #BG
        self.screen.blit(self.Background, (0,0))
        #Enemies
        for e in self.enemies:
            e.show()
        #Player
        self.player(self.playerX, self.playerY)

        if not self.game_over:
            self.show_score(self.textX, self.textY)
            self.show_ammo(8,40)
            #if self.ammo_value <1.9:
                #warning = self.font.render("Last Chance!!! Low amunition!!!", True, (255, 0, 0))
                #self.screen.blit(warning, (200,200))

        pygame.display.flip()  

env = environment(1.5,1,1,10,6,1,1.5,100,0.5,0.5)

agent = relu3_Qagent_linearOut_dOut_l2(**hyperdict)

try:
    agent.load("agent.weights.h5")
except:
    print("Eeeeh, brutha. No weights.")

import atexit

def cleanup():
    global agent
    agent.save("agent.weights.h5")
    print("Saved weights on inturrupt")

atexit.register(cleanup)

print("")
print("")
print("")
print("")

CURSOR_UP_ONE = '\x1b[1A' 
ERASE_LINE = '\x1b[2K' 

for i in range(number_of_train_y_bits):
    for j in range(time_step_threshold):
        state = env.state()
        action = agent.act(state)
        reward, state, newstate, done = env.step(action)

        agent.remember(state, action, reward, newstate, done)
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE) 
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE) 
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE) 
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE) 
        print(f"Training step: {i}/{number_of_train_y_bits}\nAction step: {j}/{time_step_threshold}\n{(i/number_of_train_y_bits+(j/time_step_threshold)/number_of_train_y_bits)*100}%")
    agent.replay()


#Optimise replay
#Check out sum trees
#Check out reward scaling
#Use numpy more for optimisation?
#Try caching TD errors
    
#When you have time go through T1 and II to try and get rid of hard coded shapes