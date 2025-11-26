hyperdict = {
            "gamma": 0.996,           # Discount factors for future rewards
            "layer1_size": 1024,         # Sizes of the first hidden layer
            "layer2_size": 512,         # Sizes of the second hidden layer
            "layer3_size": 256,          # Sizes of the third hidden layer
            "layer4_size": 128,           # Sizes of the fourth hidden layer
            "batch_size": 64,      # Batch sizes to try
            "learning_rate": 0.001, # Learning rates for the optimizer
            "dropout1": 0.3,          # Dropout rates for the first layer
            "dropout2": 0.2,          # Dropout rates for the second layer
            "dropout3": 0.1,         # Dropout rates for the third layer
            "reg1": 0.0001,              # L2 regularization strengths for layer 1
            "reg2": 0.0001,                # L2 regularization strengths for layer 2
            "reg3": 0.0001,                # L2 regularization strengths for layer 3
            "memory": 100000,      # Sizes of the replay memory
            "input_shape": (1, 23),         # Input shape for the model
            "actions": 4,                  # Number of possible actions (e.g., in a reinforcement learning task)
    }

from collections import deque
import numpy as np
import random as rand
from keras import layers, Sequential, regularizers, optimizers
import pygame

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

    class enemy():
        def __init__(self, parent):
            self.enemyX = rand.randint(0,750)
            self.enemyY=rand.randint(120, 200)
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
                self.parent.bullet_state="ready"
                self.parent.bulletY=730
                self.parent.score_value += 1
                self.parent.ammo_value+= self.parent.ammo_inc
                self.enemyX = rand.randint(0,750)
                self.enemyY = rand.randint(100, 200)
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
                if self.ammo_value < 1:
                    self.reset()


    def state(self):
        return np.array(
        [self.playerX, self.playerXD, 1 if self.bullet_state == "ready" else 0, self.bulletX if self.bullet_state=="fire" else 0, 
         self.bulletY if self.bullet_state =="fire" else 0] +
        [value for e in self.enemies for value in (e.enemyX, e.enemyY, e.enemyXD)]
        ).reshape(1,1,23)

    def step(self, action):

        """Takes an action and advanes the game by one step."""

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

        if self.ammo_value <1 and self.bullet_state == "ready":
            self.game_over=True

        for e in self.enemies:
            e.move()
        self.move_player()
        self.move_bullet()

        if self.game_over:
            self.reset()

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

        pygame.display.flip()  


    def __init__(self, ammo_inc, Player_Speed, Enemy_Speed, starting_ammo, num_enem, SCREEN_HEIGHT=800, SCREEN_WIDTH=850):
        
        self.rendering = False
        
        self.ammo_inc = ammo_inc
        self.Player_Speed = Player_Speed
        self.Enemy_Speed = Enemy_Speed
        self.ammo_value = starting_ammo
        self.starting_ammo = starting_ammo
        self.num_enem = num_enem
        self.SCREEN_HEIGHT=SCREEN_HEIGHT
        self.SCREEN_WIDTH=SCREEN_WIDTH

        self._initialize_pygame()

        self.reset()

class relu3_Qagent_linearOut_dOut_l2():

    def __init__(self, gamma, layer1_size, 
                 layer2_size, layer3_size, layer4_size, batch_size, learning_rate,
                 dropout1, dropout2, dropout3, reg1, reg2, reg3, memory, input_shape, actions):
        self.gamma = gamma 
        self.batch_size = batch_size
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
                  metrics=['mae']),
                  

        return model

    def load(self, path):
        if not path.endswith(".weights.h5"):
            path+=".weights.h5"

        self.model.load_weights(path)

    def act(self, state):
        return np.argmax(self.model.predict(state, verbose=0)[0])

env = environment(ammo_inc=1.5,
                  Player_Speed=1,
                  Enemy_Speed=1,
                  starting_ammo=10,
                  num_enem=6)

env.reset()
env.initialize_rendering()

b0b = relu3_Qagent_linearOut_dOut_l2(**hyperdict)

b0b.load("agent.weights.h5")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        if event.type == pygame.KEYDOWN:
            if event.key==pygame.K_SPACE:
                s = env.state()
                print(s)
                print(b0b.model.predict(s))
            if event.key==pygame.K_UP:
                env.bullet_state = "fire"

    env.step(b0b.act(env.state()))

    env.render()