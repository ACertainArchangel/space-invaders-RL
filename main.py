"""
Deep Space - DQN Agent for Space Invaders
==========================================
Training Configuration:
- Set TRAINING=True, RENDERING=False for fast training
- Set TRAINING=False, RENDERING=True to watch trained agent
- LOADING=True loads previous weights
"""

TRAINING = True
RENDERING = True   # Disable for faster training (you have to on colab cus their rendering is a pain)
EPISODES = 5000    # Number of episodes to train
MAX_STEPS = 10000  # Max steps per episode

LOADING = False    # Loed [sic]
SAVING = True
SAVE_EVERY = 100   # Save weights every N episodes

SCREEN_HEIGHT = 800
SCREEN_WIDTH = 850

from Environment import Environment
from Agent import Agent
from Preprocessor import Preprocessor
from tqdm import tqdm
import numpy as np
import pygame
import sys

# Environment with tuned reward parameters
env = Environment(
    ammo_inc=1.5,           # Ammo gained per hit
    Player_Speed=2,         # Faster player movement
    Enemy_Speed=1,
    starting_ammo=10,
    num_enem=6,
    ammo_penalty=0.1,       # Small penalty for shooting (was 1)
    hit_reward=10.0,        # Big reward for hitting (was 1.5)
    death_penalty=50.0,     # Big penalty for dying (was 100)
    closeness_penalty=0.1,  # Reduced (was 0.5)
    closeness_threshold=0.5,
    SCREEN_HEIGHT=SCREEN_HEIGHT,
    SCREEN_WIDTH=SCREEN_WIDTH
)

preprocessor = Preprocessor(SCREEN_HEIGHT, SCREEN_WIDTH)

agent = Agent(
    gamma=0.99,
    epsilon=1.0 if TRAINING else 0.0,  # No exploration when testing
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    learning_rate=0.0001,
    target_update_freq=1000
)

if LOADING:
    try:
        agent.load_model_weights("model.weights.h5")
        print("Loaded model weights successfully!")
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Starting with fresh weights.")

if RENDERING:
    env.initialize_rendering()

# Training metrics
episode_rewards = []
best_reward = float('-inf')

for episode in tqdm(range(EPISODES), desc="Training"):
    env.reset()
    state = env.state()
    processed_state = preprocessor.preprocess(state)
    
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        action = agent.select_action(processed_state, training=TRAINING)
        reward, _, newstate, done = env.step(action)
        
        newstate_processed = preprocessor.preprocess(newstate)
        episode_reward += reward

        if TRAINING:
            agent.store_experience(processed_state, action, reward, newstate_processed, done)
            agent.train_model()

        processed_state = newstate_processed

        if RENDERING:
            # Process pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
            env.render()

        if done:
            break
    
    # Decay epsilon after each episode
    if TRAINING:
        agent.decay_epsilon()
    
    episode_rewards.append(episode_reward)
    
    # Print progress every 50 episodes
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"\nEpisode {episode+1} | Avg Reward (last 50): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save best model and periodic saves
    if SAVING:
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model_weights("model_best.weights.h5")
        
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_model_weights("model.weights.h5")
            print(f"\nSaved checkpoint at episode {episode+1}")

if SAVING:
    agent.save_model_weights("model.weights.h5")
    print("\nTraining complete! Model saved.")