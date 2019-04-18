#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:51:36 2019

@author: clytie
"""

import tensorflow as tf
from algorithms.dqn import DQN


class DoubleDQN(DQN):
    def __init__(self, n_action, dim_ob_image,
                 rnd=1,
                 discount=0.99,
                 epsilon_schedule=lambda x: max(0.1, (1e5-x) / 1e5),
                 update_target_freq=5000,
                 lr=2.5e-4,
                 max_grad_norm=5,
                 save_path="./double_dqn_log",
                 save_model_freq=1000,
                 log_freq=100):
        super().__init__(n_action=n_action, 
                         dim_ob_image=dim_ob_image,
                         rnd=rnd,
                         discount=discount,
                         epsilon_schedule=epsilon_schedule,
                         update_target_freq=update_target_freq,
                         lr=lr,
                         max_grad_norm=max_grad_norm,
                         save_path=save_path,
                         save_model_freq=save_model_freq,
                         log_freq=log_freq)
        
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
        
        def _model(obs):
            x = tf.divide(tf.cast(obs, tf.float32), 255.0)
            x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            return tf.layers.dense(x, self.n_action)
            
        with tf.variable_scope("main/qnet"): # need update
            self._qvals = _model(self.observation)
            
        with tf.variable_scope("main/qnet", reuse=True): # used to compute Q(s', a', w)
            self._act_qvals = tf.stop_gradient(_model(self.next_observation))
            
        with tf.variable_scope("target/qnet"): # fixed qnet
            self._target_qvals = tf.stop_gradient(_model(self.next_observation))
            
    def _build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-5)
        trainable_variables = tf.trainable_variables("main/qnet")
        
        batch_size = tf.shape(self._done)[0]
        action_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        action_q = tf.gather_nd(self._qvals, action_index)
    
        # target
        arg_act = tf.argmax(self._act_qvals, axis=1, output_type=tf.int32)
        arg_act_index = tf.stack([tf.range(batch_size), arg_act], axis=1)
        y = self._reward + self._discount * (1 - self._done) * tf.gather_nd(self._target_qvals, arg_act_index)

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
