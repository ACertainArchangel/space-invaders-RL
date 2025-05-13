An optimised version of Byzantine I:

- Modular
- Still takes a while to run
- Uses TD error prioratised replay
- Uses multiprocessing to train and evaluate multiple models in parallel
- Evaluates the performance of combinations of hyperparameters to be used later in hyperparameter tuning.

**Data is saved in `optimum.json` in this format:<br> {<string describing hyperparameters\>:<list describing total rewards accumulated in each trial during tuning\>}**