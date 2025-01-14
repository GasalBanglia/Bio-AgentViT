# Bio-AgentViT
Code repository for the paper "Bio-AgentViT: Bio-Inspired Hybrid ViT with Patch Selection For Speeding Up Training" (working title). This codebase is an extension of the repository developed by Cauteruccio et al for their paper "Speeding up Vision Transformers Through Reinforcement Learning" (https://www.semanticscholar.org/paper/Speeding-up-Vision-Transformers-Through-Learning-Cauteruccio-Marchetti/265f9f603a60f2e6291881a38032e82518fcb10a)

## Usage
The code is divided into separate modules:
<li>vit.py              -- Definition of the simpleViT architecture</li>
<li>rl_env.py           -- Definition of the reinforcement learning environment.</li>
<li>dqn_agent.py        -- The reinforcement learning patch-selection agent.</li>
<li>train_test_agent.py -- Code for training and testing the agents.</li>

The notebooks reference the classes and functions defined in the above modules to instantiate, train, and evaluate the networks. They also provide dataset-loading services.

## Model Architectures
TODO insert graphics and diagrams.