# Instructions
## Requirements
Before running the scripts, ensure the following:
1. Python dependencies are installed.
2. SUMO is installed and SUMO_PATH is specified in the environment variables.
3. `QL.py` & `Static.py` have been updated with the following:
    - `EVALUATION` set to `True` if using the evaluation intersection.
    - `PROJECT_PATH` is the absolute path to the root.

## Running
- Run `QL.py` with `python QL.py`. This will generate a Q-Table and test the optimal policy once generated.
- Run `Static.py` with `python Static.py`. This will run 10 episodes with a static policy.