from tuners.tuner import tuner
from helpers.load_json import load_json
import json
from sys import exit
import datetime

start = datetime.datetime.now()

optimisation_settings = load_json("OptoConfig.json")

tuner = tuner(maximum_capacity=optimisation_settings["maximum_capacity"])

if optimisation_settings["add_one_rendering"]:

    parallel_settings = load_json("tuners/ParallelConfig.json")
    try:
        assert not parallel_settings["run_sessions_parallel"]
    except AssertionError:
        print("You cannot run sessions in parallel and add a rendering session at the same time. Please check your settings.")
        exit(1)

    params = tuner.get_random_hyperparameters()
    params["rendering"]=True
    tuner.add(**params)
    optimisation_settings["maximum_capacity"]-=1


tuner.fill_up(gigabytes_reserved=optimisation_settings["gigabytes_reserved"], 
              cores_reserved=optimisation_settings["cpu_cores_reserved"]) #Set this to -1 to diable the cores reserved check

try:
    assert tuner.sessions!=[]
except AssertionError:
    print("No sessions were added. Please check your settings.")
    exit(1)


data = tuner.tune(action_steps=optimisation_settings["action_steps_during_training"], 
                  killers=optimisation_settings["killers"], 
                  killer_ratio=optimisation_settings["killer_ratio"], 
                  testing_steps=optimisation_settings["testing_steps"],
                  train_during_pruning=optimisation_settings["train_during_pruning"])

end = datetime.datetime.now()

writing = {"Data":data,
           "Start":str(start),
           "End":str(end),
           "Duration":str(end-start),
           "Settings":optimisation_settings,
           "Environment_Settings":load_json("tuners/EnvConfig.json")
           }

print(data)

filename = optimisation_settings["output_file"]
with open(filename, 'w') as json_file:
    json.dump(writing, json_file, indent=4)

print(f"Data has been written to {filename}")