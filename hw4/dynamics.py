import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess

        mean_obs = normalization[0]
        std_obs = normalization[1]
        mean_deltas = normalization[2]
        std_deltas = normalization[3]
        mean_action = normalization[4]
        std_action = normalization[5]

        # build the model
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ob_ph = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.ac_ph = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.label_ph = tf.placeholder(shape=[None, ob_dim], name="label", dtype=tf.float32)

        obs_normal = (self.ob_ph - mean_obs) / (std_obs + 1e-3)
        act_normal = (self.ac_ph - mean_action) / (std_action + 1e-3)
        input_ph = tf.concat([obs_normal, act_normal], axis=1)

        predict_normal = build_mlp(input_ph, ob_dim, "dynamics", n_layers=n_layers, size=size, activation=activation, output_activation=output_activation)
        self.pred_next_obs = self.ob_ph + (mean_deltas + tf.multiply(std_deltas, predict_normal))
        self.mse_loss = tf.reduce_mean(tf.square(self.pred_next_obs - self.label_ph))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.mse_loss)


    def fit(self, dataset):
        """
        Write a function to take in a dataset of
                (unnormalized)states,
                (unnormalized)actions,
                (unnormalized)next_states
        and fit the dynamics model going from normalized states, 
        normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        observations = dataset[0]
        actions = dataset[1]
        next_observations = dataset[2]
        num_data = observations.shape[0]
        losses = []

        def randomize_data(obs, act, next_obs):
            import random
            index = list(range(len(obs)))
            random.shuffle(index)
            obs = obs[index]
            act = act[index]
            next_obs = next_obs[index]
            return obs, act, next_obs

        for itr in range(self.iterations):
            i = 0
            if itr % 10 == 0: print("dynamics iter {}".format(itr))
            observations, actions, next_observations = randomize_data(observations, actions, next_observations)
            while i+self.batch_size <= num_data:
                batched_obs = observations[i:i+self.batch_size]
                batched_act = actions[i:i+self.batch_size]
                batched_next_obs = next_observations[i:i+self.batch_size]
                self.sess.run(self.train_step, feed_dict={self.ob_ph:batched_obs, self.ac_ph:batched_act, self.label_ph:batched_next_obs})
                i += self.batch_size
            if num_data % self.batch_size != 0:
                batched_obs = observations[i:]
                batched_act = actions[i:]
                batched_next_obs = next_observations[i:]
                self.sess.run(self.train_step, feed_dict={self.ob_ph:batched_obs, self.ac_ph:batched_act, self.label_ph:batched_next_obs})
            train_loss = self.sess.run(self.mse_loss, feed_dict={self.ob_ph:batched_obs, self.ac_ph:batched_act, self.label_ph:batched_next_obs})
            losses.append(train_loss)
            print("loss {}".format(train_loss))
        return np.array(losses)


    def predict(self, states, actions):
        """
        Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model
        """
        """ YOUR CODE HERE """
        states = states.reshape((-1, states.shape[-1]))
        actions = actions.reshape((-1, actions.shape[-1]))
        return self.sess.run(self.pred_next_obs, feed_dict={self.ob_ph:states, self.ac_ph:actions}).reshape(states.shape)


