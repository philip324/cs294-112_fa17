"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import numpy as np
import tensorflow as tf
import pickle
import tf_util
import gym
import load_policy
import argparse
import matplotlib.pyplot as plt
import run_expert


all_rollouts = [1,5,10,15,20,25,30,35,40,45,50]
# all_rollouts = [5,10]
network_size = 150
batch_size = 100
starter_lr = 1e-3


def load_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.loads(f.read())
    data = dataset["observations"]
    labels = dataset["actions"]
    labels = labels.reshape((labels.shape[0], labels.shape[2]))
    data, labels = randomize_data(data, labels)
    return data, labels


def randomize_data(data, labels):
    import random
    index = list(range(len(data)))
    random.shuffle(index)
    data = data[index]
    labels = labels[index]
    return data, labels


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # start training
    filename = "rollout_data/" + str(args.envname) + "_" + str(args.num_rollouts) + ".pkl"
    data, labels = load_data(filename)
    # build the model
    num_data = data.shape[0]
    input_size, output_size = data.shape[1], labels.shape[1]
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])
    hidden = tf.layers.dense(inputs=x, units=network_size, activation=tf.nn.relu)
    y_hat = tf.layers.dense(inputs=hidden, units=output_size)
    loss = tf.nn.l2_loss(y_hat - y)
    # decaying learning rate
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(starter_lr, global_step, 100000, 0.96, staircase=True)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    def training(train_data, train_labels, epoch=100, new_session=None):
        if new_session is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        else:
            sess = new_session
        losses = []
        for _ in range(epoch):
            # randomize data at the beginning of each epoch
            train_data, train_labels = randomize_data(train_data, train_labels)
            i = 0
            while i+batch_size <= num_data:
                batched_data, batched_label = train_data[i:i+batch_size], train_labels[i:i+batch_size]
                sess.run(train_step, feed_dict={x: batched_data, y: batched_label})
                i += batch_size
            if num_data % batch_size != 0:
                batched_data, batched_label = train_data[i:], train_labels[i:]
                sess.run(train_step, feed_dict={x: batched_data, y: batched_label})
            train_loss = sess.run(loss, feed_dict={x: train_data, y: train_labels})
            losses.append(train_loss)
        return sess, losses

    def run_rollouts(sess, rollouts):
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(rollouts):
                if i % 10 == 0: print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    bc_action = sess.run(y_hat, feed_dict={x: obs[None,:]})
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(bc_action)
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

    #####################
    ###  Section 3.1  ###
    #####################
    sess, losses = training(data, labels)
    means, stds = [], []
    for r in all_rollouts:
        print("{} starts.".format(r))
        returns = run_rollouts(sess, r)
        mean, std = np.mean(returns), np.std(returns)
        means.append(mean)
        stds.append(std)

    print(np.mean(means))
    print(np.mean(stds))

    #####################
    ###  Section 3.2  ###
    #####################
    # means, stds, old_e = [], [], 1
    # sess, losses = training(data, labels, epoch=1)
    # returns = run_rollouts(sess, 20)
    # mean, std = np.mean(returns), np.std(returns)
    # means.append(mean)
    # stds.append(std)
    # for e in all_rollouts[1:]:
    #     sess, losses = training(data, labels, epoch=(e-old_e), new_session=sess)
    #     returns = run_rollouts(sess, 20)
    #     mean, std = np.mean(returns), np.std(returns)
    #     means.append(mean)
    #     stds.append(std)
    #     old_e = e


    ### plot reward ###
    # plt.plot(all_rollouts, 4200*np.ones(len(all_rollouts)), color="red")
    plt.errorbar(all_rollouts, means, yerr=stds, fmt='-o', color="blue")
    plt.title("Mean returns of {} for {} rollouts".format(args.envname, args.num_rollouts))
    plt.xlabel("rollouts")
    plt.ylabel("return")
    plt.show()

    ### plot losses ###
    # t = np.r_[:len(losses)]
    # plt.plot(t, losses)
    # plt.title("losses of {} for {} rollouts".format(args.envname, args.num_rollouts))
    # plt.xlabel("rollouts")
    # plt.ylabel("losses")
    # plt.show()


if __name__ == '__main__':
    main()
