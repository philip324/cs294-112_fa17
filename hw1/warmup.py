import numpy as np
import tensorflow as tf
import pickle
import tf_util
import random
import gym
import load_policy
import argparse
import matplotlib.pyplot as plt

network_size = 100
iteration = 100
lr = 0.1

def load_data(filename):
	with open(filename, 'rb') as f:
		dataset = pickle.loads(f.read())
	data = dataset["observations"]
	labels = dataset["actions"]
	labels = labels.reshape((labels.shape[0], labels.shape[2]))
	# shuffle
	index = list(range(len(data)))
	random.shuffle(index)
	data = data[index]
	labels = labels[index]
	return data, labels


def training(train_data, train_label):
	input_size  = train_data.shape[1]
	output_size = train_label.shape[1]
	x = tf.placeholder(tf.float32, [None, input_size])
	y = tf.placeholder(tf.float32, [None, output_size])
	# build the model
	dense_lyr = tf.layers.dense(inputs=x, units=network_size, activation=tf.nn.relu)
	y_hat = tf.layers.dense(inputs=dense_lyr, units=output_size)
	loss = tf.nn.l2_loss(y_hat - y)
	optimizer = tf.train.AdamOptimizer(lr)
	train_step = optimizer.minimize(loss)
	# start training
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	losses = []
	for _ in range(iteration):
		sess.run(train_step, feed_dict={x: train_data, y: train_label})
		train_loss = sess.run(loss, feed_dict={x: train_data, y: train_label})
		losses.append(train_loss)
	return losses


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str)
	args = parser.parse_args()

	filename = "rollout_data/" + args.data + ".pkl"
	train_data, train_label = load_data(filename)
	losses = training(train_data, train_label)

	step = np.r_[:len(losses)]
	plt.plot(step, losses)
	plt.title("Training Loss of {}".format(args.data))
	plt.xlabel("iteration")
	plt.ylabel("loss")
	plt.axis([0,100,0,90000])
	plt.show()


if __name__ == '__main__':
	main()

