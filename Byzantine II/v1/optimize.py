from tuners.tuner import tuner_for_bob, get_remaining_ram_in_gb
import json

print(get_remaining_ram_in_gb())

tuner = tuner_for_bob(maximum_capacity=2)#oops this will not leave space for data. Fix that!!!

tuner.fill_up(gigabytes_reserved=0.5) #Add better verbosity mode!!!

#tuner.add_render(**tuner.get_random_hyperparameters())

assert tuner.sessions!=[]

data = tuner.tune(action_steps=10000, killers=3, killer_ratio=0.5, testing_steps=2)

filename = "optimum.json"
with open(filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)  # indent for pretty printing

print(f"Data has been written to {filename}")