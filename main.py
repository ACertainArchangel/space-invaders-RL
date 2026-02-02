TRAINING = False
RENDERING = True
LOOPS = 10000

LOADING = True
SAVING = False

SCREEN_HEIGHT = 800
SCREEN_WIDTH = 850

from Environment import Environment
from Agent import Agent
from Preprocessor import Preprocessor
from tqdm import tqdm

env = Environment(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, SCREEN_HEIGHT=SCREEN_HEIGHT, SCREEN_WIDTH=SCREEN_WIDTH)
env.initialize_rendering()

Preprocessor = Preprocessor(SCREEN_HEIGHT, SCREEN_WIDTH)

Agent = Agent()

if LOADING:
    Agent.load_model_weights("model.weights.h5")

if RENDERING:
    env.initialize_rendering()
for i in tqdm(range(LOOPS)):
    state = env.state()
    processed_image = Preprocessor.preprocess(state)
    action = Agent.select_action(processed_image)
    reward, state, newstate, done = env.step(action)

    newstate_image = Preprocessor.preprocess(newstate)

    if TRAINING:
        Agent.store_experience(processed_image, action, reward, newstate_image, done)
        Agent.train_model()

    if done:
        env.reset()

    if RENDERING:
        env.render()

if SAVING:
    Agent.save_model_weights("model.weights.h5")