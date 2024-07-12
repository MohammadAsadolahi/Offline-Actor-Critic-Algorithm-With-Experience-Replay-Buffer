# Offline-Actor-Critic-Algorithm-With-Experience-Replay-Buffer
Combining experience replay buffer whith Actor-Critic algorithm to expedite learning procedure by offline learning
This codde is a single Actor-critic Agent deployed along with a experience replay buffer which is comonly used in off-policy reinforcement algorithms  

  The intial idea to not use replay buffer with actor-critic methods like was proposed by Phil Tabor at this Youtube video: https://www.youtube.com/watch?v=LPBC3SkXwAY   
  he argues that its a bad idea i tried to make it a good one! ha ha
  this first insight made me develop the TDS algorithm, a Stochastic Off-Policy Actor-Critic Algorithm for contiuous RL problems : https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic


**this project can be applied for !any on-policy Actor-Critic algorithm! to expedite learning procedure**   
**this project is a research project and owners have no responsibility of any loss for deploying this project in real environmentn**   

### How to install requirements
The `requirements.txt` file should list all Python libraries that the project depend on, and they will be installed using:
```
pip install -r Requirements.txt
```
I keep updating the project to be compatible with new versions of libraries. If there was any problem with the diffrent versions of the required libraries let me know in the "Issues" section, so i can resolve them.  
  
  
**this code is implemented with Pytorch! i will add TensorFlow version soon and link to it here!!**  
### to do:  
* deploy the ReplayBuffer code   [***done***]
* add imports file   [***done***]
* deploy Value neural network [***done***]
* deploy Policy neural network [***done***]
* deploy the Agnet in pytorch [***partially done (Agent class methods should be ipmplemented)***]
* deploy the Environmetn class (using OpenAI Gym library to import an environment) 
* deploy the main learning loop
* deploy the test loop
* run a standard RL evaluation 
