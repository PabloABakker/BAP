# BAP
Authors: Pablo A. Bakker, Amir H. Mohammad.

This repository is created for the TU DELFT Bsc Final Project, spcifically for the subgroup DelFly control using sparse reinforcement learning techniques.
The main objective was to create a controller for a flapping wing drone (DelFly) using Reinforcement Learning, distill the knowledge using a system identification technique : SINDy.
This was done for the following reasons: the sindy algorithmn produces sparse interpretable equations, which is for obvious reasons better for implementation. Furthermore we extended the applications of the sindy algorithmn to encorporate hardare efficient objectives.

## How to install dependencies and compile the code?

- Make sure you are using is Visual Studio Code as the IDE
- Go to RL_enviroment
- Open a new terminal and make sure you are inside the RL_enviroment directory
- pip install -r requirements.txt

You should now have all the dependancies to run the whole pipeline

## How to run through the whole pipeline?
# Training the agent using RL

-Now that you are inside the RL_enviroment directory open the RL folder and change your directory path to ..\RL
-To train an agent on an enviroment type inside your terminal: python Train_and_test.py --train --env (enviroment name e.g. CartPole-v1) --algo (e.g. ppo, sac, be carefull choosing what algorithmn when, for example sac cannot be used for discrete action spaces) note that the brackets are only for explanatory purpose and should not be used inside of the terminal for example --algo ppo
-To look at the progress of your agent learning open a new terminal and go to the RL directory
-Next type inside of the terminal: tensorboard --logdir logs and click on the link in the terminal
-Tuning the hyperparameters of each algorithmn can be done inside of the 'train' function

# Gathering data from the trained agent
-After training the agent it is now time to gather data from multiple trajectories to train the sindy model
- Go to the imitation.py script inside of the rl folder and at the bottem change the arguments of the collect_data function accordingly (for example on the cartpole enviroment env_id=CartPole-v1 and algo='ppo') and run the script

# Training the sindy model on the gathered data
- Now that you have collected data from multiple trajectories off the mlp policy it is time to make it smalle
- Go back to the RL_enviroment repository and from there open the Sindy repository in your terminal
- Next open the imitate_rl_policy file and at the bothom inside of the analyze with sindy function at the data_file argument copy the path of the previously created csv file
- You should see a model now printed in the terminal and an mse, probably the mse will not be that great whitout tuning so sadly you will now have to tune it yourself by adjussting the length of the polynomial library or the fourier library and changing the learning parameter inside of the optimizer

# Finally the validation of the found model
- Inside of the Sindy repository type the following in your terminal: python test_sindy_policy.py --env enviroment --algo algorithmn --dt dt --render
- Congratulations you should now see both policies rendered on the enviroment and a comparison of the amount of parameters needed aswell as the states plottet against eachother
