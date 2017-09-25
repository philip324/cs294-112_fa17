# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.


To run warmup:

	python warmup.py Hopper-v1_100

To run section 3.1, comment out the lines corresond to section 3.2, and vice versa:

	python behavioral_cloning.py expert/HalfCheetah-v1.pkl HalfCheetah-v1 --num_rollouts 50
	python behavioral_cloning.py expert/HalfCheetah-v1.pkl HalfCheetah-v1 --num_rollouts 5

To run section 4.2:

	python DAgger.py expert/Ant-v1.pkl Ant-v1 --num_rollouts 50
