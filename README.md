# ALLAgents
This Repo builds upon the Autonomous Learning Library (ALL) (https://github.com/cpnota/autonomous-learning-library), adding the AC, CACLA, FAC, and TOCLA reinforcement learning agents.

Currently, this repo requires some changes in the ALL library (which are on the features/normalise_inputs feature branch) of the forked ALL repository

To run the cacla agent, you need:
- python3
- pip3
- clone my forked ALL repo, and switch to the features/normalise_inputs branch.
  - after cloning, cd autonomous-learning-library, and run
  ```
  pip3 install -e .
  ```
- install pytorch using `pip3 install torch torchvision`
- install tensorboard using `pip3 install tensorboard`

Then, cd into ALLAgents directory, and run:
```
python3 main.py MountainCarContinuous-v0 tocla --episodes=150
```

It should solve the MountainCarContinuous-v0 environment in around 200 episodes, and should solve the Pendulum-v0 environment in around 1200 episodes
