#Import Modules
import os

import pygame
import math
import random as rand
from keras import layers, Sequential, optimizers
import numpy as np
from collections import deque

####################################################

full_send = True

ammo_inc = 1.5

Player_Speed = 1.2

Enemy_Speed = 1.2

ammo_value = float(10)

askVals = False
        
num_enem = 6

interval = 0.4 #time step

ammo_penalty = 1

hit_reward = 1.4

batch_size = 64

death_penalty = 0

border_penalty = 0.1

saving_interval = 10

alpha = 0 # 1 => Prioratise reward 100% in training. #0 => uniform distrabution

beta = 0 #Add to the absolute value of the reward when prioratising learning to prevent 0 probability

#######################################################

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5060)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.18  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9935
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(layers.Input(shape=self.state_size))  # Specify the input shape using Input layer
        model.add(layers.Flatten())  # Flatten the input to a 1D array
        model.add(layers.Dense(128, activation='relu'))  # First hidden layer
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(self.action_size, activation='linear'))  # Output layer
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))  # Use learning_rate
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        print(f"epsilon: {self.epsilon}")

        if not full_send:
            self.epsilon=0

        if np.random.rand() <= self.epsilon:
            print("Random")
            return np.random.choice(self.action_size)  # Explore
        act_values = self.model.predict(state)
        print("Smart")
        return np.argmax(act_values[0])  # Exploit


    def replay(self, batch_size):
        #minibatch = np.random.choice(len(self.memory), batch_size)
        # Calculate priorities based on absolute reward magnitude
        priorities = np.array([abs(reward)+beta for _, _, reward, _, _ in self.memory])
    
        priorities = priorities ** alpha
    
        # Calculate probabilities
        probabilities = priorities / np.sum(priorities)
    
        # Sample minibatch indices based on probabilities
        minibatch = np.random.choice(len(self.memory), batch_size, p=probabilities)
    
        #
        for index in minibatch:
            state, action, reward, next_state, done = self.memory[index]
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.epsilon=max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)
        print("loading...")

    def save(self, name):
        self.model.save_weights(name)
        print("Saving...")

state_old = np.zeros((1, num_enem+1,3))
state = np.zeros((1, num_enem+1,3))

agent = DQNAgent((num_enem+1, 3), 4)
agent.load("model.keras.weights.h5")

def engine():
    global state, state_old
    state_old = np.array(state)

    state[0][0] = np.array([playerX, playerY, playerXD])
    for ind, (x, y, dx) in enumerate(zip(enemyX, enemyY, enemyXD)):
        state[0][ind]=np.array([x, y, dx])

######################################################

pygame.init()
game_over = False

gofont = pygame.font.Font("resources/TNR.ttf", 100)
scfont = pygame.font.Font("resources/TNR.ttf", 80)
#Create Screen
screen = pygame.display.set_mode((850, 800))
#Background
Background = pygame.image.load("resources/space.jpg")
# Title and Icon
pygame.display.set_caption("Space Invaders")
icon = pygame.image.load("resources/ufo.png")
pygame.display.set_icon(icon)


#Player Function (XD = change)
PlayerImg = pygame.image.load("resources/hero.png")
playerX = 400
playerY = 730
playerXD = 0
def player(x,y):
    screen.blit(PlayerImg, (x, y))


#Bullet - Ready = no see fire = move
bulletImg = pygame.image.load("resources/bolt.png")
bulletX = 400
bulletY = 730
bulletXD = 0
bulletYD = 20
bullet_state = "ready"
def fire_bullet(x,y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x, y))
def is_collision(enemyX, enemyY, bulletX, bulletY, coldist):
    distance = math.sqrt(((math.pow(enemyX-bulletX, 2))+(math.pow(enemyY-bulletY,2))))
    if distance < coldist:
        return True
    else:
        return False



#ENEMies
enemyImg = []
enemyX = []
enemyY = []
enemyXD = []
enemyYD = []
for i in range(num_enem):
    enemyImg.append(pygame.image.load("resources/ufo.png"))
    enemyX.append(rand.randint(0,750))
    enemyY.append(rand.randint(50, 150))
    enemyXD.append(Enemy_Speed) 
    enemyYD.append(40)
def enemy(x, y, i):
    if not game_over:
        screen.blit(enemyImg[i], (x, y))


#Smoothness variables and score
pleft=0
pright=0
score_value = 0
font = pygame.font.Font("resources/TNR.ttf", 32)
textX = 10
textY = 10
def show_score(x,y):
    score = font.render("Score: %d" %score_value, True, (255, 255, 255))
    screen.blit(score, (x, y))
def show_ammo(x,y):
    ammo = font.render("Ammo: %.1f" %ammo_value, True, (255, 255, 255))
    screen.blit(ammo, (x, y))


#Action functions
def reset():
    global pright, pleft, playerXD
    if pright == 1:
        pleft = 0
        playerXD=Player_Speed
    else:
        pleft=0
        playerXD=0
    if pleft == 1:
        pright = 0
        playerXD = -Player_Speed
    else:
        pright = 0
        playerXD =0
def left():
    global pleft, playerXD
    reset()
    playerXD = -Player_Speed
    pleft = 1
def right():
    global pright, playerXD
    reset()
    playerXD = Player_Speed
    pright = 1
def up():
    global playerXD, bulletX, ammo_value, reward, ammo_penalty
    reward -= ammo_penalty
    if playerXD>0:
        playerXD=Player_Speed
    if playerXD<0:
        playerXD=-Player_Speed
    if bullet_state == "ready" and not game_over and not ammo_value < 1:
        bulletX=playerX
        fire_bullet(playerX,playerY)
        ammo_value = ammo_value - 1 

    
















#Game Loop -- Persistant objects must be in loop!!!
running = True
frames_clock = 0
learning_clock = 0
saving_clock = 0
reward = 0

while running:
    
    #RGB - RED GREEN BLUE
    screen.fill((0,0,0))
    

    #persist background
    screen.blit(Background, (0,0))


    ##EVENTS
    for event in pygame.event.get():

        #Quit?
        if event.type == pygame.QUIT:
            running = False
        #Press
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                left()
            if event.key == pygame.K_RIGHT:
                right()
            if event.key == pygame.K_UP:
                up()
            if event.key == pygame.K_DOWN:
                reset()
            
        
            
            
    if ammo_value <1 and bullet_state == "ready":
        game_over = True

    #ENEMY MOVEMENT
    for i in range(num_enem):
        if game_over:
            break
       
        if enemyX[i]<0:
            enemyXD[i]= Enemy_Speed #0.5
            enemyY[i]=enemyYD[i]+enemyY[i]
        
        elif enemyX[i]>786:
            enemyXD[i]=-Enemy_Speed#-0.5
            enemyY[i]=enemyYD[i]+enemyY[i]
        enemyX[i] = enemyXD[i] + enemyX[i]
        
        collision = is_collision(enemyX[i], enemyY[i], bulletX, bulletY, 27) and bullet_state == "fire"
        game_over = is_collision(enemyX[i],enemyY[i],playerX,playerY, 33)
        

        if collision:
            reward +=hit_reward

            bullet_state="ready"
            bulletY=730
            score_value += 1

            ammo_value = ammo_value + ammo_inc
            enemyX[i] = rand.randint(0,750)
            enemyY[i] = rand.randint(50, 150)
            if enemyX[i]%2 == 0:
                enemyXD[i] = Enemy_Speed #0.5
            else:
                enemyXD[i] = -Enemy_Speed #0.5
            #num_enem -= 1
            #print(num_enem)
            
        

        #Call Enemy
        enemy(enemyX[i],enemyY[i], i)




    #ADD CHANGE PLAYER
    if playerX<0:
        playerX=0
    elif playerX>786:
        playerX=786
    else:
        playerX = playerXD + playerX

    #Call player creation
    player(playerX,playerY)

    
    
    #Bullet Movement
    if bullet_state == "fire":
        fire_bullet(bulletX, bulletY)
        bulletY -= bulletYD

    if bulletY < 0:
        bulletY = 730
        bullet_state = "ready"

    if not game_over:
        show_score(textX, textY)
        show_ammo(8,40)
        if ammo_value <1.9:
            warning = font.render("Last Chance!!! Low amunition!!!", True, (255, 0, 0))
            screen.blit(warning, (200,200))
    elif game_over:
        
        score = scfont.render("Your score was %d" %(score_value-1), True, (255, 255, 255))
        screen.blit(score, (150,400))
        GameOverText = gofont.render("Game Over!", True, (255,255,255))
        screen.blit(GameOverText, (150, 100))
        

    if game_over:
        reward-=death_penalty
        score_value = 0

    if playerX>800 or playerX<50:
        reward-=border_penalty


    for i in enemyY:
        if i>500:
            reward-=0.25

    #update ML
    if frames_clock>=60*interval:
        engine()
        action = agent.act(state)

        if action == 0:
            up()
        if action == 1:
            left()
        if action == 2:
            right()
        if action == 3:
            reset()

        agent.remember(state_old, action, reward, state, game_over)
        print(f"reward this block:{reward}")
        reward = 0

        frames_clock = 0

        learning_clock+=1

    if learning_clock>=batch_size:
        learning_clock = 0
        agent.replay(batch_size)
        saving_clock+=1

    if saving_clock>=saving_interval:
        saving_clock=0
        agent.save("model.keras.weights.h5")
    
    frames_clock+=1

    if game_over:
        game_over = False
        frames_clock = 0
        #ENEMies
        enemyImg = []
        enemyX = []
        enemyY = []
        enemyXD = []
        enemyYD = []
        for i in range(num_enem):
            enemyImg.append(pygame.image.load("resources/ufo.png"))
            enemyX.append(rand.randint(0,750))
            enemyY.append(rand.randint(50, 150))
            enemyXD.append(Enemy_Speed) 
            enemyYD.append(40)
        ammo_value = 10

    #Update screen
    pygame.display.update()