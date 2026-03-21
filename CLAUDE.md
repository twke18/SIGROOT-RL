# Reinforcement Learning for Robot Learning

This is a code base for training half-cheeta locomotion with reinforcement learning.

## Setup

To set up the code base, the following components are needed:

1. **Gym environment**: A simulation environment supported by Mujoco for half-cheeta robot locomotion.  Read [this website](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for the introduction of the gym environment.
2. **RL algorithms**: Reinforcement learning frameworks that train policies to complete the task.  We consider two algorithms: Proximal Policy Optimization (PPO) and Flow Matching Policy Gradients (FPO).  Read [this repo](https://github.com/akanazawa/fpo/tree/main) that implmenets both algorithms.
3. **Robot policies**: Robot policies are neural networks.  Read @./instructions/RobotPolicy.md for more context.
4. **Runner script**: A script that create environments, run RL algorithms for training, and test the performance of trained policies.  Read [rl_games](https://github.com/Denys88/rl_games) that coordinates training-testing pipeline with RL algorithms.
5. **Configuration files**: A file that contain configurable parameters for the environment, the algorithm and the training/testing pipeline.

## Dependency requirement

1. Gym environment should base on `Mujoco`.
2. RL algorithms and robot policies should be implemented with `pytorch`.
3. Runner scripts should base on `rl_games`.

Mimize the list of required dependencies.

## Trainging and testing pipeline

1. Initiate multiple environments parallely
2. Build the robot policy
3. Training
    - Collecting rollout phase: sample random actions with the current policy and interact within environments to collect state-action-reward information
    - Learning phase: train the policy with the collected rollouts using either PPO / FPO algorithm
4. Testing
    - Reset all environments to initial states
    - Rollout trajectories with the current policy
    - Report the accumulated reward and the final task success rate

## Logging results

Both training and testing results will be logged to facilitate debugging

1. Training logs include:
    - Collecting rollout phase: The average accumulated reward from all environments
    - Learning phase: The actor and critic loss
2. Testing logs include:
    - The accumulated reward and the final task success rate
    - The rendered videos of robots performing the task

All numerical results should be logged with tensorboard. All visual results should be saved locally. You should take care of naming. Users should be able to distinguish results from PPO or FPO easily.

## Suggested Workflow

1. Check the dependency requirements from `Mujoco`, `rl_games` and `pytorch`
2. Propose the best version of these packages which enable simple installation and minimal dependencies
3. Create an initial runner script at `./runner.py`. The script should support execution of both PPO and FPO algorithms. Therefore, create the corresponding configuration files in `./configurations/PPO.yaml` and `./configurations/FPO.yaml`.
4. Initiate HalfCheeta Environment from Mujoco
5. Render videos with the environment by performing random actions
6. Implement training and testing pipeline based on `rl_games`
7. Implement PPO algorithm and the Gaussian policy based on `rl_games`
8. Run PPO algorithm with the runner script
9. Verify the training and testing results of PPO algorithms
10. Implement FPO algorithm and the flow matching policy
11. Run FPO algorithm with the runner script
12. Verify the training and testing results of FPO algorithms
13. Write a user guidance file at `./README.md`