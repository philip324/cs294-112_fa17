import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env
        self.ac_dim = env.action_space.shape[0]

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return np.random.uniform(low=-1.0, high=1.0, size=self.ac_dim)


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.ac_dim = env.action_space.shape[0]

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        obs = np.resize(state, (self.num_simulated_paths, len(state)))

        observations, next_observations = [], []
        actions = np.random.uniform(low=-1.0, high=1.0, size=(self.horizon, self.num_simulated_paths, self.ac_dim))

        for i in range(self.horizon):
            observations.append(obs)
            obs = self.dyn_model.predict(obs, actions[i,::])
            next_observations.append(obs)

        observations = np.array(observations)
        next_observations = np.array(next_observations)
        cost = trajectory_cost_fn(self.cost_fn, observations, actions, next_observations)
        min_idx = np.argmin(cost)
        return actions[0,min_idx,:]

