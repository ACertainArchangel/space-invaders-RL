from trainers.trainer import trainer
import psutil
import random as rand

def get_remaining_ram_in_gb():
    # Get the available memory in bytes
    available_memory = psutil.virtual_memory().available
    # Convert bytes to GB
    return available_memory / (1024 ** 3)

class tuner_for_bob(trainer):
    def __init__(self, maximum_capacity):
        super().__init__(maximum_capacity)

        self.hyperdict = {
            "alpha": [0, 0.4, 0.7, 1],  # Learning rates to try
            "epsilon_init": [0.5, 1.0],           # Initial exploration rates
            "epsilon_decay": [0.95, 0.99, 0.999], # Decay rates for exploration
            "epsilon_min": [0.01, 0.1],           # Minimum exploration rates
            "gamma": [0.9, 0.95, 0.99, 0.995],           # Discount factors for future rewards
            "layer1_size": [1024],         # Sizes of the first hidden layer
            "layer2_size": [512],         # Sizes of the second hidden layer
            "layer3_size": [256],          # Sizes of the third hidden layer
            "layer4_size": [128],           # Sizes of the fourth hidden layer
            "batch_size": [16, 32, 64, 128],      # Batch sizes to try
            "learning_rate": [0.0001, 0.001, 0.01], # Learning rates for the optimizer
            "dropout1": [0.1, 0.2, 0.3, 0.5],          # Dropout rates for the first layer
            "dropout2": [0.1, 0.2, 0.3, 0.5],          # Dropout rates for the second layer
            "dropout3": [0.1, 0.2, 0.3, 0.5],          # Dropout rates for the third layer
            "reg1": [0.0, 0.01, 0.1],              # L2 regularization strengths for layer 1
            "reg2": [0.0, 0.01, 0.1],              # L2 regularization strengths for layer 2
            "reg3": [0.0, 0.01, 0.1],              # L2 regularization strengths for layer 3
            "memory": [1000, 10000, 100000],      # Sizes of the replay memory
            "input_shape": [(1, 23)],         # Input shape for the model
            "actions": [4],                  # Number of possible actions (e.g., in a reinforcement learning task)
            "sample_size_for_TDERR":[200],
            "time_step_threshold":[100]
    }

    def fill_up(self, gigabytes_reserved):
        created_instances = 0
        while get_remaining_ram_in_gb()>gigabytes_reserved and len(self.sessions)<self.maximum_capacity:
            self.add(**self.get_random_hyperparameters())
            created_instances+=1
        print(f"Created {created_instances} to be trained using fill_up().")

    def get_random_hyperparameters(self):
        def random_dict(input_dict):
            return {key: rand.choice(value) for key, value in input_dict.items()}
        
        print(param:=random_dict(self.hyperdict))
        
        return param
    
    def tune(self, action_steps=3000, killers=3, killer_ratio=0.5, testing_steps=1000):
        assert action_steps%killers==0

        report = {}

        for session in self.sessions:
            report[str(session.params)]=[]

        for killer in range(killers):
            steps=0

            for i in range(int(action_steps/killers)):
                self.step()
                steps+=1
                print(f"Step {steps}/{action_steps/killers} killer {killer}/{killers}")

            print(f"Pruning {len(self.sessions)} instances!")
            self.prune(killer_ratio, testing_steps)
            for session in self.sessions:
                report[str(session.params)].append(session.env.total_reward)
                session.env.reset()
            print("Finished this round of pruning!")

        return report      
        
            
    def prune(self, killer_ratio, steps):#e0 pruning
        for session in self.sessions:
            session.env.reset()
            session.agent.epsilon_save = float(session.agent.epsilon)
            session.agent.epsilon = 0.0
        
        print("Running reward gathering.")
        for i in range(steps):
            self.step()

        total_rewards = [session.env.total_reward for session in self.sessions]

        threshold_index = int(len(total_rewards) * killer_ratio)

        if threshold_index<1:  # If killer_ratio is very small, avoid index error
            print(f"Threshold index corrected to 1 from {threshold_index}")
            threshold_index = 1

        threshold_value = sorted(total_rewards)[threshold_index - 1]

        self.sessions = [session for session in self.sessions if session.env.total_reward >= threshold_value]

        for session in self.sessions:
            session.agent.epsilon = session.agent.epsilon_save

        


