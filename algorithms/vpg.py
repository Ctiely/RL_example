#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:55:34 2019

@author: clytie
"""

import numpy as np
import tensorflow as tf
from algorithms.base import Base



class VanillaPolicyGradient(Base):
    def __init__(self,
                 n_action, dim_ob_image,
                 rnd=1,
                 temperature=1.0,
                 discount=0.99,
                 lr=2.5e-4,
                 gae=0.95,
                 entropy_coef=0.01,
                 critic_coef=1.0,
                 max_grad_norm=0.5,
                 save_path="./vpg_log"):
        self.n_action = n_action
        self.dim_ob_image = dim_ob_image
        
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.tau = gae
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        self.lr = lr

        super().__init__(save_path=save_path, rnd=rnd)
    
    def _build_network(self):
        self.ob_image = tf.placeholder(
            tf.uint8, [None, *self.dim_ob_image], name="image_observation")

        x = tf.divide(tf.cast(self.ob_image, tf.float32), 255.0)
        x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)

        self.logit_action_probability = tf.layers.dense(
            x, self.n_action,
            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        self.state_value = tf.squeeze(tf.layers.dense(
            x, 1, kernel_initializer=tf.truncated_normal_initializer()))
        
    def _build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")
        
        # Get selected action index.
        batch_size = tf.shape(self.ob_image)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)
        
        log_prob = tf.nn.log_softmax(self.logit_action_probability)
        prob = tf.nn.softmax(log_prob)
        entropy = - tf.reduce_mean(log_prob * prob, axis=1)   # entropy = - 1/n \sum_i p_i \log(p_i)
        
        log_act = tf.gather_nd(log_prob, selected_action_index)
        target = - tf.reduce_mean(log_act * self.advantage)
        critic_loss = tf.reduce_mean(tf.squared_difference(self.state_value, self.target_state_value))
        
        total_loss = target - self.entropy_coef * entropy + self.critic_coef * critic_loss
        grads = tf.gradients(total_loss, tf.trainable_variables())
        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # train_op
        self._train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())
        
    def get_action(self, obs):
        batch_size = obs.shape[0]
        logit = self.sess.run(self.logit_action_probability,
                              feed_dict={self.ob_image: obs})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit / self.temperature) / np.sum(np.exp(logit / self.temperature), axis=1, keepdims=True)
        action = [np.random.choice(self.n_action, p=prob[i, :]) for i in range(batch_size)]
        return action
        
    def update(self, s_batch, a_batch, r_batch, d_batch):
        advantage_batch, target_value_batch = [], []
        for i in range(len(d_batch)):
            traj_size = len(d_batch[i])
            adv = np.empty(traj_size, dtype=np.float32)

            state_value = self.sess.run(
                self.state_value,
                feed_dict={self.ob_image: s_batch[i]})

            delta_value = r_batch[i] + self.discount * (1 - d_batch[i]) * state_value[1:] - state_value[:-1]

            last_advantage = 0

            for t in reversed(range(traj_size)):
                adv[t] = delta_value[t] + self.discount * self.tau * (1 - d_batch[i][t]) * last_advantage
                last_advantage = adv[t]

            # Compute target value.
            target_value_batch.append(state_value[:-1] + adv)
            # Collect advantage.
            advantage_batch.append(adv)
        
        # Flat the batch values.
        advantage_batch = np.concatenate(advantage_batch, axis=0)
        target_value_batch = np.concatenate(target_value_batch, axis=0)
        all_step = sum(len(dones) for dones in d_batch)

        s_batch = np.concatenate([s[:-1] for s in s_batch], axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        advantage_batch = advantage_batch.reshape(all_step)
        target_value_batch = target_value_batch.reshape(all_step)

        # Normalize Advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)
        
        # update
        self.sess.run(self._train_op,
                      feed_dict={self.ob_image: s_batch,
                                 self.action: a_batch,
                                 self.advantage: advantage_batch,
                                 self.target_state_value: target_value_batch})
