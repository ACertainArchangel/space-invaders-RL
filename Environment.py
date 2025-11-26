"""Space Invaders Reinforcement Learning Environment.

This module implements a Space Invaders game environment for training RL agents.
The environment follows a gym-like interface with step() and reset() methods.
"""

import sys
from typing import Tuple, List, Optional

class NullWriter:
    """A writer that suppresses pygame initialization output."""
    
    def write(self, arg: str) -> None:
        """Suppress write operations."""
        pass
    
    def flush(self) -> None:
        """Suppress flush operations."""
        pass
original = sys.stdout
sys.stdout=NullWriter()

#Import Modules
import pygame
import random as rand
import numpy as np

sys.stdout = original

class Environment():
    """Space Invaders game environment for reinforcement learning.
    
    This environment simulates a Space Invaders game where an agent controls a player
    that must shoot enemies while managing limited ammunition. The environment provides
    rewards for hitting enemies and penalties for running out of ammo or letting enemies
    get too close.
    
    Attributes:
        pygame_initialized (bool): Class variable tracking pygame initialization status.
    """

    pygame_initialized: bool = False

    @classmethod
    def _initialize_pygame(cls) -> None:
        """Initialize pygame if not already initialized."""
        pygame.init()
        cls.pygame_initialized = True

    @staticmethod
    def _is_collision(enemyX: float, enemyY: float, bulletX: float, bulletY: float, 
                      colision_threshold: float) -> bool:
        """Check if two objects are colliding based on distance threshold.
        
        Args:
            enemyX: X coordinate of the enemy.
            enemyY: Y coordinate of the enemy.
            bulletX: X coordinate of the bullet.
            bulletY: Y coordinate of the bullet.
            colision_threshold: Maximum distance for collision detection.
            
        Returns:
            True if distance between objects is less than threshold, False otherwise.
        """
        distance = np.linalg.norm(np.array([enemyX, enemyY]) - np.array([bulletX, bulletY]))
        return distance < colision_threshold
    
    def __init__(self, ammo_inc: float, Player_Speed: float, Enemy_Speed: float, 
                 starting_ammo: float, num_enem: int, ammo_penalty: float,
                 hit_reward: float, death_penalty: float, closeness_penalty: float, 
                 closeness_threshold: float, SCREEN_HEIGHT: int = 800, SCREEN_WIDTH: int = 850) -> None:
        """Initialize the Space Invaders environment.
        
        Args:
            ammo_inc: Amount of ammo gained per enemy hit.
            Player_Speed: Speed of player movement (pixels per step).
            Enemy_Speed: Speed of enemy movement (pixels per step).
            starting_ammo: Initial ammunition count.
            num_enem: Number of enemies in the game.
            ammo_penalty: Reward penalty for shooting (encourages ammo conservation).
            hit_reward: Reward for hitting an enemy.
            death_penalty: Penalty for game over (running out of ammo or collision).
            closeness_penalty: Penalty multiplier when enemies get too close.
            closeness_threshold: Fraction of screen height that triggers closeness penalty.
            SCREEN_HEIGHT: Height of the game window in pixels.
            SCREEN_WIDTH: Width of the game window in pixels.
        """
        self.rendering: bool = False
        
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

        self.total_reward: float = 0

    def initialize_rendering(self) -> None:
        """Initialize pygame rendering components.
        
        Loads all visual assets (fonts, images, sprites) and creates the game window.
        Must be called before render() if visualization is desired.
        """
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
        def show_score(x: int, y: int) -> None:
            """Display the current score on screen."""
            score = self.font.render("Score: %d" %self.score_value, True, (255, 255, 255))
            self.screen.blit(score, (x, y))

        def show_ammo(x: int, y: int) -> None:
            """Display the current ammo count on screen."""
            ammo = self.font.render("Ammo: %.1f" %self.ammo_value, True, (255, 255, 255))
            self.screen.blit(ammo, (x, y))

        self.show_score=show_score
        self.show_ammo=show_ammo

    def reset(self) -> None:
        """Reset the environment to its initial state.
        
        Resets player position, bullet state, enemies, and current reward.
        Does not reset total_reward to allow tracking across episodes.
        """
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

        def fire_bullet(x: float, y: float) -> None:
            """Fire a bullet from the given position."""
            self.bullet_state = "fire"
            if self.rendering:
                self.screen.blit(self.bulletImg, (x, y))
            
        self.fire_bullet=fire_bullet

        self.score_value = 0

        self.enemies=[self.enemy(self) for i in range(self.num_enem)]

        self.reward = 0

    class enemy():
        """An enemy object that moves horizontally and bounces at screen edges.
        
        Enemies move in a random horizontal direction and descend when hitting screen edges.
        They can be hit by bullets and reset to a new position when destroyed.
        """
        def __init__(self, parent: 'Environment') -> None:
            """Initialize an enemy with random starting position and direction.
            
            Args:
                parent: Reference to the parent environment instance.
            """
            self.enemyX = rand.randint(0,750)
            self.enemyY=rand.randint(50, 150)
            self.enemyXD=parent.Enemy_Speed*(rand.randint(0,1)*2-1)
            self.enemyYD=40
            self.parent = parent
            #parent.enemies.append(self)

        def show(self) -> None:
            """Render the enemy sprite at its current position."""
            self.parent.screen.blit(self.parent.enemyImg, (self.enemyX, self.enemyY))

        def move(self) -> None:
            """Update enemy position and handle collisions.
            
            Moves the enemy horizontally, bounces at screen edges, checks for
            bullet collisions and player collisions, and awards/penalizes accordingly.
            """
            if self.enemyX<0:
                self.enemyXD= self.parent.Enemy_Speed #0.5
                self.enemyY=self.enemyYD+self.enemyY
        
            elif self.enemyX>786:
                self.enemyXD=-self.parent.Enemy_Speed#-0.5
                self.enemyY=self.enemyYD+self.enemyY
            self.enemyX = self.enemyXD + self.enemyX
        
            collision = self.parent._is_collision(self.enemyX, self.enemyY, self.parent.bulletX, self.parent.bulletY, 27) and self.parent.bullet_state == "fire"
            if not self.parent.game_over:
                self.parent.game_over = self.parent._is_collision(self.enemyX, self.enemyY, self.parent.playerX, self.parent.playerY, 33)


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

    def move_player(self) -> None:
        """Update player position and enforce screen boundaries.
        
        Moves the player based on playerXD velocity and clamps position
        to stay within screen bounds.
        """
        if self.playerX<0:
            self.playerX=0
        elif self.playerX>786:
            self.playerX=786
        else:
            self.playerX = self.playerXD + self.playerX

    def move_bullet(self) -> None:
        """Update bullet position and handle bullet lifecycle.
        
        Moves the bullet upward when fired, resets it when it leaves the screen,
        and triggers game over if ammo runs out.
        """
        if self.bullet_state == "fire":
            self.fire_bullet(self.bulletX, self.bulletY)
            self.bulletY -= self.bulletYD

            if self.bulletY < 0:
                self.bulletY = 730
                self.bullet_state = "ready"
                if self.ammo_value < 1:
                    self.game_over = True

    def phobia(self) -> None:
        """Apply penalties when enemies get too close to the player.
        
        Penalizes the agent proportionally to how far past the closeness threshold
        each enemy has descended on the screen.
        """
        for e in self.enemies:
            if e.enemyY>self.SCREEN_HEIGHT*self.closeness_threshold:
                self.reward-=self.closeness_penalty*e.enemyY/self.SCREEN_HEIGHT
                self.total_reward-=self.closeness_penalty*e.enemyY/self.SCREEN_HEIGHT


    def state(self) -> np.ndarray:
        """Get the current state of the environment as a numpy array.
        
        Returns:
            A numpy array of shape (1, 1, 23) containing:
            - Player position and velocity (3 values)
            - Bullet state and position (3 values)
            - Enemy positions and velocities (3 values per enemy * num_enem)
        """
        return np.array(
        [self.playerX, self.playerXD, 1 if self.bullet_state == "ready" else 0, self.bulletX if self.bullet_state=="fire" else 0, 
         self.bulletY if self.bullet_state =="fire" else 0] +
        [value for e in self.enemies for value in (e.enemyX, e.enemyY, e.enemyXD)]
        ).reshape(1,1,23)

    def step(self, action: int) -> Tuple[float, np.ndarray, np.ndarray, bool]:
        """Execute one time step within the environment.
        
        Args:
            action: Action to take. 0=no move, 1=move left, 2=move right, 3=shoot.
            
        Returns:
            A tuple containing:
            - reward (float): Reward received for this step.
            - state (np.ndarray): State before the action.
            - newstate (np.ndarray): State after the action.
            - done (bool): Whether the episode has ended.
        """

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

    def render(self) -> None:
        """Render the current game state to the screen.
        
        Draws the background, enemies, player, and HUD elements.
        Requires initialize_rendering() to be called first.
        """
        self.screen.fill((0,0,0))

        self.screen.blit(self.Background, (0,0))

        for e in self.enemies:
            e.show()

        self.player(self.playerX, self.playerY)

        if not self.game_over:
            self.show_score(self.textX, self.textY)
            self.show_ammo(8,40)

        pygame.display.flip()  

    def get_picture_as_numpy(self) -> np.ndarray:
        """Get the current rendered screen as a numpy array.
        
        Returns:
            A numpy array representing the current screen pixels.
        """

        if not self.rendering:
            raise RuntimeError("Rendering not initialized. Call initialize_rendering() first.")
        
        return np.array(pygame.surfarray.pixels3d(self.screen))

if __name__=="__main__":
    env = Environment(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850)
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
                sys.exit(0)

        react = env.step(action)
        if react[3]==True:
            print(react)
        env.render()