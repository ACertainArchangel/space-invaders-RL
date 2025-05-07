## Contributing to Byzantine

Byzantine is a fun project that I work on by myself, but I would be very grateful for any contributions! Whether you see a place for optimization or want to document your own trials with different hyperparameter combinations, your input is welcome.

## What should a contribution look like?

A new contribution should be in the form of a new version or model:

- **New Versions**: New versions should be named `v<n+1>` if the last version was `v<n>`. 
- **New Models**: If you create a new model, it's yours and it can be called whatever you want!

Where to place contributions:

- **New Versions**: Versions go in their model folders next to all the other versions.<br>
- **New Models**: Models go right in the root directory (`space_inv_repo`)


## Guidelines for if a contribution should be a version or a model.

### New versions:
- If you adjust the hyperparameters of a model that accepts hyperparameters, and run it, it's a new version.
- If you make a minor change to how the environment works, how rewards are shaped, how hyperparameter data is collected, it's a new version.
- If you make a minor optimisation, it's a new version.
- New versions should include a `results.md` file documenting how the changes made improved performance if at all ***and*** a `version_info.md` file detailing the changes made. If the version was never run you don't need a `results.md`.

### New models:
- If you made a major overhaul to the internal workings of an agent (e.g. switched from Keras to Pytorch) it's a new model.
- If you added new functionality that will drastically change the user experience (e.g. added new .py files) it's a new model.

There may be some gray area where changes could either be a model or a version, but if you document what the changes are it will be fine. Again, this is an informal project.

### Questions?
If you have any questions about contributing to the project feel free to reach out and ask. 

Thank you for reading!
