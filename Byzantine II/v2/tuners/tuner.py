import random as rand
from joblib import Parallel, delayed
from helpers.load_json import load_json
from helpers.get_remaining_ram import get_remaining_ram_in_gb
from agents.agent import relu3_Qagent_linearOut_dOut_l2 as agent
from environments.env import environment as env
import os

class tuner():
    """A tuner for the agent. It is used to tune the hyperparameters of the agent using a parallel training process."""
    def __init__(self, maximum_capacity):
        self.maximum_capacity=maximum_capacity
        self.sessions=[]

        self.hyperdict = load_json("tuners/TunerConfig.json")
        self.hyperdict['input_shape'][0] = tuple(self.hyperdict['input_shape'][0])

        self.EnvKwargs = load_json("tuners/EnvConfig.json")

        self.ParallelSettings = load_json("tuners/ParallelConfig.json")

        self.DebugSettings = load_json("tuners/DebugConfig.json")
    
    def _printalt(self, content):
        if self.DebugSettings["verbose"]:
            print(content)

    class session():
        """A session is a single instance of the environment and agent. It is used to train the agent in the environment."""
        def __init__(self, agent, env, time_step_threshold, params, rendering=False):
            self.agent = agent
            self.env = env
            self.time_step = 0
            self.time_step_threshold=time_step_threshold
            self.params = params
            self.rendering = rendering

            if rendering:
                self.env.initialize_rendering()
                print("Rendering initialized.")

        def step(self, inhibit_training=False):
            """Performs a step in the environment using the agent's action."""
            action = self.agent.act(self.env.state())
            reward, state, newstate, game_over = self.env.step(action)

            if not inhibit_training:
                self.agent.remember(state, action, reward, newstate, game_over)
                self.time_step+=1

                if self.time_step > self.time_step_threshold:
                    self.agent.replay()
                    self.time_step = 0

            if self.rendering:
                self.env.render()

    def train(self, steps, print_every=1000, inhibit_training=False):
        """Trains the models in parallel for a given number of steps. If inhibit_training is True, the agent will not train during the steps."""
        def run_session(session, steps):
            for i in range(steps):
                session.step(inhibit_training=inhibit_training)
                if i % print_every == 0:
                    self._printalt(f"Session {hash(str(session.params))} step {i}/{steps} - Epsilon: {session.agent.epsilon} - Total reward: {session.env.total_reward}")

            if not inhibit_training: 
                session.agent.replay()
            return session
        
        if not inhibit_training:
            self._printalt(f"Training {len(self.sessions)} models with {steps} action steps each.")
        else:
            self._printalt(f"Training {len(self.sessions)} models with {steps} action steps each.")

        if self.ParallelSettings["run_sessions_parallel"]:
            self.sessions = Parallel(n_jobs=-1)(delayed(run_session)(session, steps) for session in self.sessions)
        else:
            for session in self.sessions:
                run_session(session, steps)
            

    def add(self, **kwargs): ##Rendering is false by default because of the try-except block
        """Adds a new session to the trainer. The session is created with the given hyperparameters. To render set kwargs["rendering"]=True; rendering is False by default.
        Takes the kwargs needed by an agent plus `time_step_threshold` and `rendering`. Rendering is False by default and time_step_threshold is required."""

        try:
            rendering = kwargs["rendering"]
            del kwargs["rendering"]
        except KeyError:
            rendering = False

        try:
            time_step_threshold = kwargs["time_step_threshold"]
            del kwargs["time_step_threshold"]
        except KeyError:
            raise Exception("time_step_threshold is a required parameter.")

        if len(self.sessions)<self.maximum_capacity or self.maximum_capacity==-1:
            self.sessions.append(self.session(env=env(**self.EnvKwargs), agent=agent(**kwargs), params=kwargs, rendering=rendering,
                                              time_step_threshold=time_step_threshold))
            return 0
        else:
            return -1
        

    def fill_up(self, gigabytes_reserved, cores_reserved=0):
        """Fills the trainer with sessions until the maximum capacity is reached or the available RAM is less than the reserved amount."""
        created_instances = 0
        while get_remaining_ram_in_gb()>gigabytes_reserved and len(self.sessions)<self.maximum_capacity and (os.cpu_count()>cores_reserved or cores_reserved<0):
            self.add(**self.get_random_hyperparameters())
            created_instances+=1
        self._printalt(f"Created {created_instances} to be trained using fill_up().")

    def get_random_hyperparameters(self):
        """Returns a random set of hyperparameters from the hyperparameter dictionary of possible values."""
        def _random_dict(input_dict):
            return {key: rand.choice(value) for key, value in input_dict.items()}
        
        self._printalt(param:=_random_dict(self.hyperdict))
        
        return param
    
    def tune(self, action_steps=3000, killers=3, killer_ratio=0.5, testing_steps=1000, train_during_pruning=True):
        """Returns a dictionary with the parameters and their respective rewards after tuning.
        action_steps: The number of action steps to take during training total (must be devisable by killers).
        killers: The number of killing cycles to use for pruning.
        killer_ratio: The ratio of models to keep after pruning. (e.g. 1=keep all, 0.5=keep half, 0.1=keep 10%)
        testing_steps: the number of steps used to test the models during pruning."""

        assert action_steps%killers==0

        report = {}

        for session in self.sessions:
            report[str(session.params)]={"rewards":[]}

        for killer in range(killers):

            for session in self.sessions:
                report[str(session.params)]["survival"]=(action_steps*(killer+1)/killers)

            self.train(int(action_steps/killers))

            self._printalt(f"Pruning {len(self.sessions)} instances!")

            session_strings = [str(session.params) for session in self.sessions]

            rewards = self.prune(killer_ratio, testing_steps, train=train_during_pruning)

            for i in range(len(session_strings)):
                report[session_strings[i]]["rewards"].append(rewards[i])
                #session.env.total_reward = 0
                session.env.reset()
            self._printalt("Finished this round of pruning!")

            for session in self.sessions:
                report[str(session.params)]["survival"]=(action_steps*(killer+1)/killers)

        return report      
        
            
    def prune(self, killer_ratio, steps, train):#e0 pruning
        """Prunes the models based on their performance. The worst performing models are removed."""
        
        old_number = len(self.sessions)
        self._printalt(f"Pruning {old_number} models to {float(old_number*killer_ratio)}-ish (parallel running)")

        def reset_job(session):
            session.env.reset()
            session.agent.epsilon_save = float(session.agent.epsilon)
            session.agent.epsilon = 0.0
            session.env.total_reward = 0
            return session
        if self.ParallelSettings["reset_jobs_parallel"]:
            self.sessions = Parallel(n_jobs=-1)(delayed(reset_job)(session) for session in self.sessions)
        else:
            for session in self.sessions:
                reset_job(session)
        
        self._printalt("Running reward gathering.")
        self.train(steps, inhibit_training = not train)

        total_rewards = [session.env.total_reward for session in self.sessions]

        threshold_index = int(len(total_rewards) * killer_ratio)

        if threshold_index<1:  # If killer_ratio is very small, avoid index error
            self._printalt(f"Threshold index corrected to 1 from {threshold_index}")
            threshold_index = 1

        threshold_value = sorted(total_rewards, reverse=True)[threshold_index - 1]

        self._printalt(threshold_value)

        self.sessions = [session for session in self.sessions if session.env.total_reward >= threshold_value]

        self._printalt(f"Pruned to {len(self.sessions)} models from {old_number}")

        def set_job(session):
            session.agent.epsilon = session.agent.epsilon_save
            return session
        if self.ParallelSettings["set_jobs_parallel"]:
            self.sessions = Parallel(n_jobs=-1)(delayed(set_job)(session) for session in self.sessions)
        else:
            for session in self.sessions:
                set_job(session)
        self._printalt("Finished reset. Resuming.")

        return total_rewards
