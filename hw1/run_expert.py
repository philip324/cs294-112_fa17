#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import behavioral_cloning

def run_rollouts(args, rollouts=None, bc=False):
    if rollouts is None:
        rollouts = args.num_rollouts

    if bc:
        filename = "rollout_data/" + str(args.envname) + "_" + str(rollouts) + ".pkl"
        train_data, train_label = behavioral_cloning.load_data(filename)
        input_size, output_size  = train_data.shape[1], train_label.shape[1]
        x = tf.placeholder(tf.float32, [None, input_size])
        y = tf.placeholder(tf.float32, [None, output_size])
        dense_lyr = tf.layers.dense(inputs=x, units=behavioral_cloning.network_size, activation=tf.nn.relu)
        y_hat = tf.layers.dense(inputs=dense_lyr, units=output_size)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                if bc:
                    bc_action = sess.run(y_hat, feed_dict={x: obs[None,:]})
                    obs, r, done, _ = env.step(bc_action)
                else:
                    obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        # with open("rollout_data/{}_{}.pkl".format(args.envname, args.num_rollouts), "wb") as output_file:
        #     pickle.dump(expert_data, output_file)
        return returns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    run_rollouts(args)


if __name__ == '__main__':
    main()

