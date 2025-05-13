from environments.env import environment as env
from agents.archetecture import relu3_Qagent_linearOut_dOut_l2 as agent
import psutil
from joblib import Parallel, delayed

def get_remaining_ram_in_gb():
    # Get the available memory in bytes
    available_memory = psutil.virtual_memory().available
    # Convert bytes to GB
    return available_memory / (1024 ** 3)

class trainer:
    def __init__(self, maximum_capacity):
        self.maximum_capacity=maximum_capacity
        self.sessions=[]

    class session():
        """A session is a single instance of the environment and agent. It is used to train the agent in the environment."""
        def __init__(self, agent, env, time_step_threshold, params, rendering=False):
            self.agent = agent
            self.env = env
            self.total_reward = 0
            self.time_step = 0
            self.time_step_threshold=time_step_threshold
            self.params = params

            if rendering:
                self.env.initialize_rendering()
            self.rendering = rendering

        def step(self):
            """Performs a step in the environment using the agent's action."""
            action = self.agent.act(self.env.state())###############################################
            reward, state, newstate, game_over = self.env.step(action)
            self.agent.remember(state, action, reward, newstate, game_over)
            self.time_step+=1
            if self.rendering:
                self.env.render()

        def train(self):
            """Trains the agent using the experience replay."""
            self.agent.replay()

    def train(self, steps, print_every=1000):
        """Trains the models in parallel for a given number of steps."""
        def run_session(session, steps):
            for i in range(steps):
                session.step()
                if i % print_every == 0:
                    print(f"Session {id(session)} step {i}/{steps} - Epsilon: {session.agent.epsilon} - Total reward: {session.env.total_reward}")
            return session
        
        print(f"Starting a parallel training process for {len(self.sessions)} models with {steps} action steps each. This will take a while.")
        self.sessions = Parallel(n_jobs=-1)(delayed(run_session)(session, steps) for session in self.sessions)
            

    def add(self, **kwargs):
        """Adds a new session to the trainer. The session is created with the given hyperparameters."""
        if len(self.sessions)<self.maximum_capacity or self.maximum_capacity==-1:
            self.sessions.append(self.session(env=env(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850), agent=agent(**kwargs), time_step_threshold=kwargs["time_step_threshold"], params=kwargs, rendering=False))
            return 0
        else:
            return -1
        
    def add_render(self, **kwargs):
        """Adds a new session to the trainer that renders the environment. The session is created with the given hyperparameters."""
        if len(self.sessions)<self.maximum_capacity or self.maximum_capacity==-1:
            self.sessions.append(self.session(env=env(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850), agent=agent(**kwargs), time_step_threshold=kwargs["time_step_threshold"], params=kwargs, rendering=True))
            return 0
        else:
            return -1
        
        



        


            