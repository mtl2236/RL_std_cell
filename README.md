# RL_std_cell

This program is an reforcement learning program to train an agent that can automaticly generate valid flexible standard cells based on seven parameters(VDD, Mu, Vth, SS, Cox, RCS, RCD) from a compact model(Athor:Leilai Shao). First of all, the generated standard cell must have normal logic function. Besides, the PDP(Power Delay Product) is another optimization target which lower is better.

The structure of this whole program is the same as AutoCkt. The reward is extracted from .lib file generated by Cadence Liberate. Due to License constraint, we can temporarily run 5 simulation threads parallelly. 

The reward in each step is defined as follows:
if a cell is failed, its equivalent PDP is defined as -10; If a cell is succeed, its PDP is defined as its average PDP of all the possible logic flips. A step's reward is calculated as the average equivalent PDP of each cell in the library. Reward is defiend as success ratio minus average equivalent PDP of standard cell library(ie. 100-1.1=98.9). If reward is larger than 99, stpe reward is replaced by an extra reward 1000.  

The equivalent PDP is defined as follows:
For all the combinational standard cells, they can be defined as a combination of M NAND2 gates. Therefore, to "normalize" standard cells with different complexity, its equivalent PDP can be defined as its original PDP divided by M. 

Up to now, the total steps are set as 2400, an episode contains at most 30 steps(episode terminates in advance when extra reward is get), the policy neural network is updated every 240 steps. Each episode begins at the worst initial parameters(lowest VDD, largest Vth). The average reward of each episode is shown below. In the last iteration, the agent can successfully generate standard cells whose average PDP less than 1 in each episode.

![episode_reward_mean](https://user-images.githubusercontent.com/89757542/192124943-de6337c5-7f2c-4e67-b9d9-8496c5c38d4b.png)
![episode_len_mean](https://user-images.githubusercontent.com/89757542/192124944-c9926386-58cc-40a0-8c39-f114df67251e.png)


Useage:
python autockt/val_autobag_ray.py
