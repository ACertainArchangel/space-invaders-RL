### v1 -> v2:
* Fixed a MAJOR bug that caused models to not train during tuning
* Added different configurations for parallelization like `run_sessions_parallel`, `set_jobs_parallel`, `reset_jobs_parallel` so I can test which is most efficient.
* Cleaned up the code
* Added config files
* Fixed hard coded shapes in agent.py **(TO DO)**
* Merged trainer and tuner
* Changed confusing file names
* Changed `optimum.json` structure to accommodate for stopping times.
    - which means `optimum.json` records early killed models too.
* Made it so that tuning does not end on pruning but rather more training and adjusted logs accordingly **TO DO**