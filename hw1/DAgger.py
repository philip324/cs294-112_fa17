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
    index = list(range(data.shape[0]))
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

    def training(train_data, train_labels, epoch=100, sess=None):
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        num_data = train_data.shape[0]
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

    def run_rollouts(sess, rollouts, expert=False):
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(rollouts):
                if i % 5 == 0: print('rollouts', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    if expert:
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                    else:
                        bc_action = sess.run(y_hat, feed_dict={x: obs[None,:]})
                        actions.append(bc_action)   # change
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
            return returns, expert_data

    iterations = 10
    rollouts = 20
    epochs = 5
    old_data, old_labels = data.copy(), labels.copy()

    # DAgger policy
    print("DAgger policy")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dagger_means, dagger_stds = [], []
    for i in range(iterations):
        print("iter",i)
        sess, losses = training(data, labels, epoch=epochs, sess=sess)
        returns, policy_data = run_rollouts(sess, rollouts)
        dagger_means.append(np.mean(returns))
        dagger_stds.append(np.std(returns))
        new_data = policy_data["observations"]
        new_labels = policy_data["actions"]
        new_labels = new_labels.reshape((new_labels.shape[0], new_labels.shape[2]))
        data = np.vstack((data, new_data))
        labels = np.vstack((labels, new_labels))


    # behaviorial cloning policy
    print("behaviorial cloning policy")
    sess.run(tf.global_variables_initializer())
    sess, losses = training(old_data, old_labels)
    bc_agent_means, bc_agent_stds = [], []
    for i in range(iterations):
        print("iter",i)
        returns, _ = run_rollouts(sess, rollouts)
        bc_agent_means.append(np.mean(returns))
        bc_agent_stds.append(np.std(returns))


    ### plot reward ###
    t = np.r_[:iterations]
    Ant = 4800
    expert_plot, = plt.plot(t, Ant*np.ones(iterations), color="red", label="expert")
    bc_plot = plt.errorbar(t, bc_agent_means, yerr=bc_agent_stds, fmt='-o', color="blue", label="bc")
    dagger_plot = plt.errorbar(t, dagger_means, yerr=dagger_stds, fmt='-o', color="green", label="DAgger")
    plt.title("Mean returns of {} for {} rollouts".format(args.envname, args.num_rollouts))
    plt.xlabel("iterations")
    plt.ylabel("return")
    plt.legend(handles=[expert_plot, bc_plot, dagger_plot])
    plt.show()


if __name__ == '__main__':
    main()
