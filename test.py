#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:44:49 2019

@author: jiachenlu
"""
import Sparse_2 as sp
import numpy as np
import tensorflow as tf
import os

n = 25
sparsity = 0.064
A = sp.genSpar(n, sparsity)
adj, xadj = sp.upperSpar(A)
adj, xadj = sp.getConSpar(adj, xadj)
a0 = 0
separator = sp.findNeighbour(adj, xadj, a0)

length = len(xadj)-1
learning_rate = 0.95
f = open("test.txt", "w")

def preprocess(separator, adj, xadj):
    zeros = [0 for i in range(length)]
    states = np.array([zeros,zeros,zeros])
    state = []
    for i in range(n):
        if i in separator:
            state.append(1)
        else:
            state.append(0)
    states[0][:len(state)] = state
    states[1] = adj
    states[2][:len(xadj)] = xadj
    return states.reshape(1, n_input, length)

n_input = 3 #separator, adj, xadj
n_action = n
n_hidden = int(2*sparsity*n**2)
n_output = int(n_action)
hidden_activation = tf.nn.relu
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(x_state, scope):
    prev_layers = [0]*n_input
    for i in range(n_input):
        prev_layers[i] = x_state[:,i,:]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        outputs = [0]*n_input
        for i in range(n_input):
            hidden = tf.contrib.layers.fully_connected(
                prev_layers[i], n_hidden, activation_fn=hidden_activation,
                weights_initializer=initializer)
            outputs[i] = tf.contrib.layers.fully_connected(
                hidden, n_output, activation_fn=hidden_activation,
                weights_initializer=initializer)
        middle_layer = np.sum(outputs)
        hidden_1 = tf.contrib.layers.fully_connected(
                middle_layer, n_output, activation_fn=hidden_activation,
                weights_initializer=initializer)
        hidden_2 = tf.contrib.layers.fully_connected(
                hidden_1, n_output, activation_fn=hidden_activation,
                weights_initializer=initializer)
        hidden_3 = tf.contrib.layers.fully_connected(
                hidden_2, n_output, activation_fn=hidden_activation,
                weights_initializer=initializer)
        output = tf.contrib.layers.fully_connected(
                hidden_3, n_output, activation_fn=None,
                weights_initializer=initializer)
        separator = x_state[:,0,:n_output]
        print(np.shape(separator))
        new_output= np.multiply(output, separator)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                                       for var in trainable_vars}
    print(trainable_vars_by_name)
    return new_output, trainable_vars_by_name


x_state = tf.placeholder(tf.float32, shape=[None,n_input, length])
actor_q_values, actor_vars = q_network(x_state, scope="q_network/actor")
critic_q_values, critic_vars = q_network(x_state, scope="q_network/critic")

copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

x_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(x_action, n_output),
                        axis=1, keep_dims=True)
#q_values = tf.squeeze(q_value)

y = tf.placeholder(tf.float32, shape=[None,1])
cost = tf.reduce_mean(tf.square(y-q_value))
global_step = tf.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

    
n_steps = 10000
training_start = 1000
training_interval = 5
save_steps = 50
copy_steps = 25
discount_rate = 0.70
batch_size = 50
iteration = 0
checkpoint_path = "./my_test.ckpt"
safe_adj = adj.copy()
print(len(safe_adj))
safe_xadj = xadj.copy()

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    while True:
        step= global_step.eval()

        iteration += 1
        
        state = preprocess(separator, adj, xadj)
        q_values = actor_q_values.eval(feed_dict={x_state: state})
        action = np.argmax(q_values)
        
        adj = safe_adj.copy()
        xadj = safe_xadj.copy()
        next_separator, reward = sp.step(
                action, adj, xadj, separator)
        adj = safe_adj.copy()
        xadj = safe_xadj.copy()
        next_state = preprocess(next_separator, adj, xadj)
        
        separator = next_separator
        
        f.write(str(action))
        f.write(' ')
        f.write(str(separator))
        f.write(' ')
        f.write(str(reward))
        f.write('\n')

        print(step)

f.close()