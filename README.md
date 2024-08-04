# ModelBasedFootstepPlanning-IROS2024
This repository is an open-sourced code for the following paper presented at the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

#### Title: Integrating Model-Based Footstep Planning with Model-Free Reinforcement Learning for Dynamic Legged Locomotion <br/> 
Paper Link: TBD <br/> 
Video Link: [https://youtu.be/Z0E9AKt6RFo](https://youtu.be/Z0E9AKt6RFo)

### Installation ###
1. Create a new python virtual env with python 3.8 using Anaconda
2. Clone this repo
```bash
git clone https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024.git humanoidGym
```
4. Install humanoidGym Requirements:
```bash
pip install -r requirements.txt
```
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 3 should still work) from https://developer.nvidia.com/isaac-gym
     - Extract the zip package
     - Copy the `isaacgym` folder, and place it in a new location
   - Install `issacgym/python` requirements
   ```bash
   cd <issacgym_location>/python
   pip install -e .
   ```
5. Install humanoidGym
    - go back to the humanoidGym repo, and install it.
    ```bash
    pip install -e .
    ```
---
## User Manual ##
## 1. Linear Inverted Pendulum Model (LIPM) Animation 
All the LIP model-related code is in the `LIPM` folder.
These codes are modified from [BipedalWalkingRobots](https://github.com/chauby/BipedalWalkingRobots) for the Center of Mass (CoM) velocity tracking task.

By running the code below, you should be able to get the following videos:
```bash
python LIPM/demo_LIPM_3D_vt.py
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/edff8522-b9d5-42d3-80af-37c0f0d50758">
</div>

By running the code below, you should be able to get the following videos and images:
```bash
python LIPM/demo_LIPM_3D_vt_analysis.py
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/2e4e1600-2b34-4181-aaea-63e2288e85e7">
  <img width = "60%" src="https://github.com/user-attachments/assets/236682a4-8beb-4e46-b2a5-e4fe76a71978">
  <img width = "60%" src="https://github.com/user-attachments/assets/bd0f33c4-7ba8-4403-9647-2d1e61091263">
</div>

## 2. MIT Humanoid in IsaacGym
### Train ###  
```bash
python gym/scripts/train.py --task=humanoid_controller
```
-  To run on CPU add the following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
-  To run headless (no rendering) add `--headless`.
- **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in `gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
-  The following command line arguments override the values set in the config files:
 - --task TASK: Task name.
 - --resume:   Resume training from a checkpoint
 - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
 - --run_name RUN_NAME:  Name of the run.
 - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
 - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
 - --num_envs NUM_ENVS:  Number of environments to create.
 - --seed SEED:  Random seed.
 - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
 - --original_cfg:  Use configs stored in the saved files associated with the loaded policy instead of the current one in envs.

### Play (a trained policy) ###  
```bash
python gym/scripts/play.py --task=humanoid_controller
```
- By default the loaded policy is the last model of the last run of the experiment folder.
- Other runs/model iteration can be selected by setting `--load_run` and `--checkpoint`.
- You would need around 3,000 iterations of training to obtain a well-behaved policy.

<div align="center">
  <img src="https://github.com/user-attachments/assets/52acb865-057a-48bf-8b6c-4ec7f3415bca">
</div>

## 3. Deploy the policy to robot hardware
This repository does not include a code stack for deploying a policy to MIT Humanoid hardware.
Please check the [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software) for our lab's hardware code stack.

To deploy the trained policy, you would need to set `EXPORT_POLICY=TRUE` in the `humanoidGym/scripts/play.py` script.
Then you would get a `policy.onnx` file to run on C++ code.

---
### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH`

---
### Acknowledgement ###
We would appreciate it if you would cite it in academic publications:
TBD
