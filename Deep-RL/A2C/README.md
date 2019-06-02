# A2C implementation in PyTorch
## Atari Pong game

This is a A2C implementation using the Gym environment Pong (Any frame based envcould be run on this). The implentation uses mulitple Virtual Environments that is bootstrapped to batch the model. The implementation also uses framestack of 4. The NN architecture is a CNN similiar to the original DQN paper. 