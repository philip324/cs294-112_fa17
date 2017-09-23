import numpy as np
import tensorflow as tf
import pickle
import tf_util
import random
import gym
import load_policy
import argparse
import matplotlib.pyplot as plt

train_test_ratio = 0.9
epoch = 50
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
	# divide into train and test
	cutoff = int(train_test_ratio * len(data))
	train_data = data[:cutoff]
	test_data  = data[cutoff:]
	train_label = labels[:cutoff]
	test_label  = labels[cutoff:]

	return train_data, test_data, train_label, test_label


def training(train_data, train_label, name):
	input_size  = train_data.shape[1]
	output_size = train_label.shape[1]
	x = tf.placeholder(tf.float32, [None, input_size])
	y = tf.placeholder(tf.float32, [None, output_size])

	# build the model
	dense_lyr = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu)
	y_hat = tf.layers.dense(inputs=dense_lyr, units=output_size)
	loss = tf.nn.l2_loss(y_hat - y)
	optimizer = tf.train.AdamOptimizer(lr)
	train_step = optimizer.minimize(loss)

	# start training
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	losses = []
	for e in range(epoch):
		sess.run(train_step, feed_dict={x: train_data, y: train_label})
		train_loss = sess.run(loss, feed_dict={x: train_data, y: train_label})
		losses.append(train_loss)

	step = np.r_[:len(losses)]
	# print(losses)
	plt.plot(step, losses)
	plt.title("Training Loss of {}".format(name))
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.show()



def main():
	filename = "rollout_data/Hopper-v1_100.pkl"
	train_data, test_data, train_label, test_label = load_data(filename)

	training(train_data, train_label, filename)
	# with open("experts/Hopper-v1.pkl", 'rb') as f:
	# 	dataset = pickle.loads(f.read())
	# print(dataset["GaussianPolicy"]["hidden"]["FeedforwardNet"]["layer_0"]["AffineLayer"]["W"].shape)
	# print(dataset["GaussianPolicy"]["hidden"]["FeedforwardNet"]["layer_2"]["AffineLayer"]["W"].shape)




if __name__ == '__main__':
	main()