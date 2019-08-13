#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:12:41 2019

@author: jiachenlu
"""


import Sparse_2 as sp
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque


# constant for the sparse matrix
n = 25 # number of vertices 
sparsity = 0.5 #sparasity
A = [[1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1]] #the 5*5 grid
adj, xadj = sp.upperSpar(A) # CVS vectors of the matrix
sp.printSparse(adj, xadj) # display the matrix
separator = {1,5} # the starting separator
length = len(adj) # the number of edges

# constant for the DQN
n_input = 3 # separator, adj, xadj
n_action = n # the size of action space
n_hidden = int(2*sparsity*n**2) # the size of each hidden layer
n_output = int(n_action) # the size of output layer
hidden_activation = tf.nn.relu #the activation function for hidden layer
initializer = tf.contrib.layers.variance_scaling_initializer() #the initializer for all the parameters  
learning_rate = 0.95 # learning rate for the second DQN

# constant for memory batching
replay_memory_size = 1000 # the maximum strorage for the replay_memory
replay_memory = deque([], maxlen=replay_memory_size) # initialize replay_memory

# constant for epsilon_greedy function
eps_min = 0.1 # the ending propbability to take a random action
eps_max = 1.0 # the starting probability to take a random action
eps_decay_steps = 1000 # the steps it need for epsilon to decrease from max to min

# constant for the training
n_steps = 1500 #total steps
training_start = 100 # when the critic DQN start learning
training_interval = 5 # the critic DQN does the learning orperation every * steps
save_steps = 50 # the parameters are saved every * steps
copy_steps = 25 # the parameters are passed from critic DQN to actor DQN every * step
discount_rate = 0.70 # the discount rate for q value
batch_size = 50 # the learning sample size
iteration = 0 # iteration counter
saver = tf.train.Saver() # saver for all parameters
checkpoint_path = "./my_dqn.ckpt" #the path to save all the parameters
saver1 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "q_network/actor")) #saver for parameters in actor DQN
test_path = "./my_test.ckpt" # the path to save the parameters for testing
safe_adj = adj.copy() # a static copy for adj
safe_xadj = xadj.copy() # a static copy of xadj
iteration_v = np.arange(1500) # x axis for plotting
q_v = [0]*1500 # y axis for plotting, record the maximun q value for each interation
init = tf.global_variables_initializer() # initializer



f = open("data.txt", "w")
f.write(str(adj))
f.write(str(xadj)+'\n')


def preprocess(separator, adj, xadj):
    # combine the adj, xadj, and separator into a 3*length matrix to fit in the placeholder
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

def q_network(x_state, scope):
    # the architecture of DQN
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
        ''' print(np.shape(separator))'''
        new_output= np.multiply(output, separator)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                                       for var in trainable_vars}
    return new_output, trainable_vars_by_name

def sample_memories(batch_size):
    # sampling memory for critic DQN to learn from
    indices = np.random.permutation(min(len(replay_memory),10*batch_size))[:batch_size]
    cols = [[], [], [], []] #state, action, reward, next_state
    for idx in indices:
        memory = replay_memory[len(replay_memory)-idx-1]
        for col, value in zip(cols,memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1),
            cols[3])

def epsilon_greedy(q_values, step, separator):
    # add random action for the network
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        f.write('random')
        return np.random.randint(0, n_action)
    else:
        return np.argmax(q_values)

x_state = tf.placeholder(tf.float32, shape=[None,n_input, length])
actor_q_values, actor_vars = q_network(x_state, scope="q_network/actor")
critic_q_values, critic_vars = q_network(x_state, scope="q_network/critic")

copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

x_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(x_action, n_output),
                        axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None,1])
cost = tf.reduce_mean(tf.square(y-q_value))
global_step = tf.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)


with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    while True:
        step= global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        
        state = preprocess(separator, adj, xadj)
        q_values = actor_q_values.eval(feed_dict={x_state: state})
        action = epsilon_greedy(q_values, step, separator)
        
        adj = safe_adj.copy()
        xadj = safe_xadj.copy()
        next_separator, reward = sp.step(
                action, adj, xadj, separator)
        adj = safe_adj.copy()
        xadj = safe_xadj.copy()
        next_state = preprocess(next_separator, adj, xadj)
        replay_memory.append((state,
                              action,
                              reward,
                              next_state))
        
        if reward != 0:
            for i in range(1,8):
                n_turn = i%4
                n_reflect = (i-n_turn)/4
                
                add_separator = {sp.add(n_turn, n_reflect, ver) 
                    for ver in separator}
                add_next_separator = {sp.add(n_turn, n_reflect, ver) 
                    for ver in next_separator}
                add_action = sp.add(n_turn, n_reflect, action)
                                
                add_state = preprocess(add_separator, adj, xadj)
                add_action = add_action
                add_reward = reward
                add_next_state = preprocess(add_next_separator, adj, xadj)

        separator = next_separator
        
        f.write(str(action))
        f.write(' ')
        f.write(str(separator))
        f.write(' ')
        f.write(str(reward))
        f.write('\n')
        
        if iteration < training_start or iteration % training_interval != 0:
            continue
        
        x_state_val, x_action_val, rewards, x_next_state_val = sample_memories(
                batch_size)
        x_state_val = tf.squeeze(x_state_val)
        x_state_val = x_state_val.eval()
        x_next_state_val = tf.squeeze(x_next_state_val)
        x_next_state_val = x_next_state_val.eval()
        next_q_values = actor_q_values.eval(
                feed_dict={x_state: x_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + discount_rate * max_next_q_values
        for i in range(len(rewards)):
            if rewards[i] == 0:
                y_val[i] = max_next_q_values[i]
            else:
                y_val[i] = rewards[i] + discount_rate * max_next_q_values[i]
            
        critic_q_values.eval(feed_dict={x_state: x_state_val})

        training_op.run(feed_dict={x_state: x_state_val,
                                   x_action: x_action_val,
                                   y: y_val})
        print(step)
        q_v[step] = max(y_val)
        plt.plot(iteration_v, q_v)
        plt.yscale("log")
        plt.show()
        if step % copy_steps == 0:
            copy_critic_to_actor.run()
        
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)
            saver1.save(sess, test_path)
f.close()