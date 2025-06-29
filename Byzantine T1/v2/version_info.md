### T1 v1 -> T1 v2:
* Adopted B0B v2's tester (lightweight and simple)
* {"dropout1": 0.3,<br>
   "dropout2": 0.2,<br>
   "dropout3": 0.1,     
   "reg1": 0.0001,              
   "reg2": 0.0001,                
   "reg3": 0.0001}
* He Initialization
* Deleted directories that didn't need to be there (oops)

Hypothesis before running:
TD error proritisation in T1 v1 caused higher Q-value value magnitudes (all in the negitive direction) by dredging up negivive memories for the model (like death and missed shots), and the dropout and L2 changes made Q-values decrece in magintude from B0B v1 to v2 (for some reason). If the B0B decrece was random noise, this will probably not perform diferently to T1 v1; if there is something to the decrece, maybe the Q-values will reduce in magnitude here. I do not excpect this version to work but maybe I will find something that can help me get performance out of one of these models. Again, as mentioned in the version info for B0B v2, I can try these in v3 of T1 too:
* Serious overhaul of reward shaping
* Massive improvement to replay.