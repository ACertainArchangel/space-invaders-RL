from tuners.tuner import tuner_for_bob
import numpy as np

tuner = tuner_for_bob(maximum_capacity=1)#oops this will not leave space for data. Fix that!!!

tuner.fill_up(gigabytes_reserved=0.5)#Add better verbosity mode!!!

assert tuner.sessions!=[]

for session in tuner.sessions:
    for i in range(100000):
        session.agent.remember(np.array([[[1.0]*23]]), 2, 10.0, np.array([[[1.1]*23]]), False)
    session.agent.replay()
    session.agent.save("hello")