#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:21:55 2019

@author: clytie
"""

import numpy as np
import tensorflow as tf
from algorithms.base import Base



class PolicyGradient(Base):
    def __init__(self,
                 n_action, dim_ob_image,
                 rnd=1,
                 temperature=1.0,
                 discount=0.99,
                 lr=2.5e-4,
                 max_grad_norm=0.5,
                 save_path="./pg_log"):
        self.n_action = n_action
        self.dim_ob_image = dim_ob_image
        
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        self._discount = discount
        self._lr = lr

        super().__init__(save_path=save_path, rnd=rnd)
        
    def _build_network(self):
        self.ob_image = tf.placeholder(
            tf.uint8, [None, *self.dim_ob_image], name="image_observation")
        self._action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self._return = tf.placeholder(shape=[None], dtype=tf.float32, name="return")
        
        x = tf.divide(tf.cast(self.ob_image, tf.float32), 255.0)
        x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        self._p_act = tf.layers.dense(x, self.n_action, activation=tf.nn.softmax)
        
    def _build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-5)
        
        batch_size = tf.shape(self._action)[0]
        log_p_act = tf.log(tf.gather_nd(self._p_act, tf.stack([tf.range(batch_size), self._action], axis=1)))

        target = - tf.reduce_mean(log_p_act * self._return)
        
        grads = tf.gradients(target, tf.trainable_variables())
        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # train_op
        self._train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()))

    def get_action(self, obs):
        batch_size = obs.shape[0]
        p_act = self.sess.run(self._p_act,
                              feed_dict={self.ob_image: obs})
        action = [np.random.choice(self.n_action, p=p_act[i, :] / self.temperature) for i in range(batch_size)]
        return action
        
    def update(self, s_batch, a_batch, r_batch):
        s_batch = np.concatenate([s[:-1] for s in s_batch], axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        reward_to_go_batch = []
        
        batch_size = len(r_batch)
        for i in range(batch_size):
            rews = r_batch[i]
            n = len(rews)
            # compute reward to go
            rtgs = np.zeros_like(rews)
            for i in reversed(range(n)):
                rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0) * self._discount
            reward_to_go_batch.append(rtgs)
        
        reward_to_go_batch = np.concatenate(reward_to_go_batch, axis=0)
        
        self.sess.run(self._train_op,
                      feed_dict={self.ob_image: s_batch,
                                 self._action: a_batch,
                                 self._return: reward_to_go_batch})
      
  