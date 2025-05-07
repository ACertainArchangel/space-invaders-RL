# Byzantine: A DQN Agent for Space Invaders

### What is this?
This is a reinforcement learning (DQN) project where I develop an agent that can play space invaders. I decided to call it "Byzantine" because it kind of sounds like "Bayesian" as in Bayesian Optimisation and it sounds kind of cool. 

### Environment:
To run a model you should be in an environment with all the dependencies installed. Create a new environment if you do not want to install the dependencies globally. These are in `requirements.txt`. You can install these dependencies by running<br>`pip install -r requirements.txt`<br>or<br>`pip3 install -r requirements.txt`<br>depending on your version of python. You should run this with space_inv_repo as the current working directory.

### Working Directory:
You should be inside the version level folder as your current working directory (e.g. `v1`, `v3`) when you run .py files. An example of a version level file path: `space_inv_repo/B0B/v1`

All .py files meant to be run are directly inside their version level folders (no extra nesting), and they are the only .py files directly in these directories.

There may be additional instructions inside version_info.md (for example about separate .py files for hyperparameter tuning,
training, and evaluation.)

### Versioning and models:
Changes between versions of the same model are minor. Any major changes will result in a new model. "Versions" just use different hyperparameters 
combinations and may have minor tweaks to the code, which will be documented. 

- Each version that has been run/tested  before will have a `results.md` file that shows how it performs and that includes data on how running went.
- Any additional useful information on a specific version will be in a `version_info.md` file.
- All versions will have ***at least one*** of these two markdown files.
- All models will have a `model_info.md` file containing model information and any special instructions on running the model.

### Note on redundancy (informational only):
Each version file is self contained.
I know each version uses the same resources and sometimes the same modules --sometimes leading to redundancy-- but the resources are very 
lightweight. Since I sometimes tweak modules between versions, the organizational payout of this redundancy tradeoff is well worth it to me.