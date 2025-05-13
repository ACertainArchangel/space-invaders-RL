#Fix bullet render

import sys
class NullWriter:
    def write(self, arg):
        pass
    def flush(self):
        pass
original = sys.stdout
sys.stdout=NullWriter()

#Import Modules
import pygame
import random as rand
import numpy as np

sys.stdout = original

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
        
        print("If this is printed more than two times than you have a serious problem!!!")
        
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
        """Resets the environment to it's initial state but does not reset total reward."""
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
        """An enemy object that moves in a random direction and can be hit by the bullet."""
        def __init__(self, parent):
            self.enemyX = rand.randint(0,750)
            self.enemyY=rand.randint(50, 150)
            self.enemyXD=parent.Enemy_Speed*(rand.randint(0,1)*2-1)
            self.enemyYD=40
            self.parent = parent
            #parent.enemies.append(self)

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

        if self.game_over:
            self.reset()

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

        return (reward, state, newstate, self.game_over)

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
#time step, rewards, input, auto reset

if __name__=="__main__":
    env = environment(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850)
    env.initialize_rendering()

    

    while True:

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_UP]:
            action = 3
        else:
            action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        react = env.step(action)
        if react[3]==True:
            print(react)
        env.render()