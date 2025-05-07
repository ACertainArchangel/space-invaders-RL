from environments.env import environment as env
from agents.archetecture import relu3_Qagent_linearOut_dOut_l2 as agent
import psutil

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
            action = self.agent.act(self.env.state())###############################################
            reward, state, newstate, game_over = self.env.step(action)
            self.agent.remember(state, action, reward, newstate, game_over)
            self.time_step+=1
            if self.rendering:
                self.env.render()

        def train(self):
            self.agent.replay()

    def step(self):
        for session in self.sessions:
            session.step()
            if session.time_step>=session.time_step_threshold:
                session.time_step=0
                session.train()

    def train(self, minisodes):

        """Unused to train a collection of models. Maybe use
        later to train all promising hyperparameter models and later save them...
        I would more likely implement that manually on the tuner superclass
        but its nice to have this..."""

        for i in range(minisodes*self.time_step):
            self.step()
            print(f"training {i}/{minisodes*self.time_step}")

    def add(self, **kwargs):
        if len(self.sessions)<self.maximum_capacity or self.maximum_capacity==-1:
            self.sessions.append(self.session(env=env(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850), agent=agent(**kwargs), time_step_threshold=kwargs["time_step_threshold"], params=kwargs, rendering=False))
            return 0
        else:
            return -1
        
    def add_render(self, **kwargs):
        if len(self.sessions)<self.maximum_capacity or self.maximum_capacity==-1:
            self.sessions.append(self.session(env=env(1.5, 1, 1, 10, 6, 1, 1.5, 100, 0.5, 0.5, 800, 850), agent=agent(**kwargs), time_step_threshold=kwargs["time_step_threshold"], params=kwargs, rendering=True))
            return 0
        else:
            return -1
        
        



        


            