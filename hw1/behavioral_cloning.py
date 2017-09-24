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

all_rollouts = [5,10,20,50,100,150,200]
network_size = 150
batch_size = 100
epoch = 50
starter_learning_rate = 1e-3

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


def training(train_data, train_label):
	num_data = train_data.shape[0]
	input_size  = train_data.shape[1]
	output_size = train_label.shape[1]
	x = tf.placeholder(tf.float32, [None, input_size])
	y = tf.placeholder(tf.float32, [None, output_size])
	# build the model
	dense_lyr = tf.layers.dense(inputs=x, units=network_size, activation=tf.nn.relu)
	y_hat = tf.layers.dense(inputs=dense_lyr, units=output_size)
	loss = tf.nn.l2_loss(y_hat - y)
	# decaying learning rate
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
											   100000, 0.96, staircase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_step = optimizer.minimize(loss)
	# start training
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	losses = []
	for _ in range(epoch):
		# randomize data at the beginning of each epoch
		train_data, train_label = randomize_data(train_data, train_label)
		i = 0
		while i+batch_size <= num_data:
			batched_data, batched_label = train_data[i:i+batch_size], train_label[i:i+batch_size]
			sess.run(train_step, feed_dict={x: batched_data, y: batched_label})
			i += batch_size
		if num_data % batch_size != 0:
			batched_data, batched_label = train_data[i:], train_label[i:]
			sess.run(train_step, feed_dict={x: batched_data, y: batched_label})
		
		train_loss = sess.run(loss, feed_dict={x: train_data, y: train_label})
		losses.append(train_loss)
	return losses



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
	args = parser.parse_args()

	filename = "rollout_data/" + str(args.envname) + "_" + str(args.num_rollouts) + ".pkl"
	train_data, train_label = load_data(filename)
	losses = training(train_data, train_label)

	means, stds = [], []
	# t = (10*np.arange(40)+10).tolist()
	for r in all_rollouts:
		print("{} starts.".format(r))
		# get returns
		returns = run_expert.run_rollouts(args, rollouts=r, bc=True)
		mean, std = np.mean(returns), np.std(returns)
		means.append(mean)
		stds.append(std)

	plt.plot(all_rollouts, means, "blue")
	plt.plot(all_rollouts, stds, "green")
	# plt.title("Mean returns of behavioral cloning agent")
	# plt.xlabel("rollouts")
	# plt.ylabel("return")
	# plt.axis([0,len(losses),0,500])
	plt.show()


if __name__ == '__main__':
	main()

