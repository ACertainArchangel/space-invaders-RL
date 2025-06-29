# To do list:

### Byzantine II v2:
* Cache session ID's so you aren't constantly recomputing a hash.
* Change how pruning works so it ends in training rather than pruning.
* Fix how model survival is documented and test further.

### Byzantine T1 v2:
* Run and document

### General:
* Check out sum trees
* __Check out reward scaling__
* Use numpy more for optimisation?

### All (v3):
* Leaky ReLU
* Fix bullet rendering
* Fix hard coded shapes in agent.py like files
* Seriously overhaul reward shaping
* Prioritise replay based on reward magnitude maybe? Forget replayed memories once they are remembered n times? I think replay can be upgraded, and this may be worthy of it's own model -> T2.

### VERSIONS:
* v1 **DONE**
    * B0B DONE
    * T1 DONE
    * II DONE
* v2
    * B0B DONE
    * T1 to be run
    * II in progress
* v3
    * B0B yet to start
    * T1 yet to start
    * II yet to start