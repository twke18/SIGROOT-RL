# RobotPolicy

This file briefly describes the architecture of robot policies.

## Architecture

Robot policies are often neural networks. In this project, we consider simple designs--multilayer perceptron (MLP)--as the backbone architecture for the policy.  These models takes as input the state / observation retrieved from the Gym environment, while outputing future actions.

The forms of output actions dependend on individual RL algorithms:
- For PPO, the policy predicts the mean and standard deviation of a Gaussian, and sample **one** future action from the Gaussian.
- For FPO, the policy generates **multiple** future actions using diffusion / flow matching mechanism.

Read [this directory](https://github.com/akanazawa/fpo/tree/main/gridworld/models) for similar design choices.