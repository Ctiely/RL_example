#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:48:23 2019

@author: clytie
"""

import numpy as np
import tensorflow as tf
from collections import deque
from algorithms.base import Base


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, trajs):
        s_batch, a_batch, r_batch, d_batch = trajs
        for states, actions, rewards, dones in zip(s_batch, a_batch, r_batch, d_batch):
            len_traj = len(dones)
            for i in range(len_traj):
                self.buffer.append((states[i], actions[i], rewards[i], dones[i], states[i + 1]))
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(range(buffer_size),
                                 size = batch_size,
                                 replace = False)
        batch = [self.buffer[i] for i in index]
        states_mb = np.asarray([each[0] for each in batch])
        actions_mb = np.asarray([each[1] for each in batch])
        rewards_mb = np.asarray([each[2] for each in batch]) 
        dones_mb = np.asarray([each[3] for each in batch])
        next_states_mb = np.asarray([each[4] for each in batch])

        return states_mb, actions_mb, rewards_mb, dones_mb, next_states_mb


class DQN(Base):
    def __init__(self, n_action, dim_ob_image,
                 rnd=1,
                 discount=0.99,
                 epsilon_schedule=lambda x: max(0.1, (8e4-x) / 8e4),
                 update_target_freq=5000,
                 lr=2.5e-4,
                 max_grad_norm=5,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=100):
        self.n_action = n_action
        self.dim_ob_image = dim_ob_image
        
        self._discount = discount
        self._update_target_freq = update_target_freq
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._max_grad_norm = max_grad_norm

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)
        
    def _build_network(self):
        # s_t
        self.observation = tf.placeholder(
            tf.uint8, [None, *self.dim_ob_image], name="cur_observation")
        # s_{t+1}
        self.next_observation = tf.placeholder(
            tf.uint8, [None, *self.dim_ob_image], name="next_observation")
        # a_t
        self._action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        # r_t
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        # d_t
        self._done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        
        with tf.variable_scope("main/qnet"): # need update
            x = tf.divide(tf.cast(self.observation, tf.float32), 255.0)
            x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self._qvals = tf.layers.dense(x, self.n_action)
            
        with tf.variable_scope("target/qnet"): # fixed target
            x = tf.divide(tf.cast(self.next_observation, tf.float32), 255.0)
            x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self._target_qvals = tf.layers.dense(x, self.n_action)

    def _build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-5)
        trainable_variables = tf.trainable_variables("main/qnet")
        
        batch_size = tf.shape(self._done)[0]
        action_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        action_q = tf.gather_nd(self._qvals, action_index)

        # target
        y = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * tf.reduce_max(self._target_qvals, axis=1))
        # loss
        loss = tf.reduce_mean(tf.squared_difference(y, action_q))
        
        # clip gradients
        grads = tf.gradients(loss, trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        # update qnet
        self._train_op = self.optimizer.apply_gradients(zip(clipped_grads, trainable_variables))
        
        def _update_target(qnet, fixed_net):
            params1 = tf.trainable_variables(qnet)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(fixed_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops
        
        # assign qnet to fixed_net
        self._update_target_op = _update_target("main/qnet", "target/qnet")
        self._log_op = {"loss": loss}
        
    def get_action(self, obs):
        batch_size = obs.shape[0]
        q = self.sess.run(self._qvals, feed_dict={self.observation: obs})
        max_a = np.argmax(q, axis=1)
        
        # epsilon greedy strategy
        global_step = self.sess.run(tf.train.get_global_step())
        actions = np.random.randint(self.n_action, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self._epsilon_schedule(global_step)
        actions[idx] = max_a[idx]
        return actions

    def update(self, databatch, sw_dir):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch

        # update
        self.sess.run(self._train_op,
                      feed_dict={
                              self.observation: s_batch,
                              self._action: a_batch,
                              self._reward: r_batch,
                              self._done: d_batch,
                              self.next_observation: next_s_batch})

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._update_target_freq == 0:
            self.sess.run(self._update_target_op)

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self.observation: s_batch,
                                    self._action: a_batch,
                                    self._reward: r_batch,
                                    self._done: d_batch,
                                    self.next_observation: next_s_batch})
            self.sw.add_scalars(sw_dir, log, global_step=global_step)
            

