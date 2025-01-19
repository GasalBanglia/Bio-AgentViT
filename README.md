# Bio-AgentViT
Code repository for the paper "Bio-AgentViT: Bio-Inspired Hybrid ViT with Patch Selection For Speeding Up Training" (working title). This codebase is an extension of the repository developed by Cauteruccio et al for their paper "Speeding up Vision Transformers Through Reinforcement Learning" (https://www.semanticscholar.org/paper/Speeding-up-Vision-Transformers-Through-Learning-Cauteruccio-Marchetti/265f9f603a60f2e6291881a38032e82518fcb10a)

## Usage
The code is divided into separate modules:
- **vit.py**              -- Definition of the simpleViT architecture
- **rl_env.py**           -- Definition of the reinforcement learning environment.
- **dqn_agent.py**        -- The reinforcement learning patch-selection agent.
- **train_test_agent.py** -- Code for training and testing the agents.

The notebooks reference the classes and functions defined in the above modules to instantiate, train, and evaluate the networks. They also provide dataset-loading services.

## Model Architectures
![image](https://github.com/user-attachments/assets/6ae318f7-f4db-416c-ac7a-ecfb685d532d)
