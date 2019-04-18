#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:07:32 2019

@author: clytie
"""

import math
import numpy as np
import tensorflow as tf
from algorithms.base import Base
from tqdm import tqdm


class PPO(Base):
    def __init__(self,
                 n_action, dim_ob_image,
                 rnd=1,
                 temperature=1.0,
                 discount=0.99,
                 gae=0.95,
                 vf_coef=1.0,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 train_epoch=10,
                 batch_size=64,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 clip_schedule=lambda x: max(0.2, (1 - x)) * 0.5,
                 save_path="./ppo_log"):
        self.n_action = n_action
        self.dim_ob_image = dim_ob_image
        
        self.temperature = temperature
        self.discount = discount
        self.tau = gae
        self.entropy_coefficient = entropy_coef
        self.critic_coefficient = vf_coef
        self.max_grad_norm = max_grad_norm
        self.training_epoch = train_epoch
        self.training_batchsize = batch_size

        self.lr_schedule = lr_schedule
        self.clip_schedule = clip_schedule
        
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
        self.clip_epsilon = tf.placeholder(tf.float32)
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)

        # \pi_\theta_old
        self.old_logit_action_probability = tf.placeholder(tf.float32, [None, self.n_action])
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")
        
        # Get selected action index.
        batch_size = tf.shape(self.ob_image)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)

        # Compute entropy of the action probability.
        log_prob_1 = tf.nn.log_softmax(self.logit_action_probability)
        log_prob_2 = tf.stop_gradient(tf.nn.log_softmax(self.old_logit_action_probability))

        prob_1 = tf.nn.softmax(log_prob_1)
        self.entropy = - tf.reduce_mean(log_prob_1 * prob_1, axis=1)   # entropy = - 1/n \sum_i p_i \log(p_i)
        
        # Compute ratio of the action probability.
        logit_act1 = tf.gather_nd(log_prob_1, selected_action_index)
        logit_act2 = tf.gather_nd(log_prob_2, selected_action_index)
        self.ratio = tf.exp(logit_act1 - logit_act2)
        
        # Get surrogate object.
        surrogate_1 = self.ratio * self.advantage
        surrogate_2 = tf.clip_by_value(self.ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage
        self.surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))
        
        # Compute critic loss.
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.state_value, self.target_state_value))

        # Compute gradients.
        self.total_loss = self.surrogate + self.critic_coefficient * self.critic_loss - self.entropy_coefficient * self.entropy
        grads = tf.gradients(self.total_loss, tf.trainable_variables())

        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # train_op
        self.total_train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())

    def get_action(self, obs):
        batch_size = obs.shape[0]
        logit = self.sess.run(self.logit_action_probability,
                              feed_dict={self.ob_image: obs})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit / self.temperature) / np.sum(np.exp(logit / self.temperature), axis=1, keepdims=True)
        action = [np.random.choice(self.n_action, p=prob[i, :]) for i in range(batch_size)]
        return action

    def _generator(self, data_batch, batch_size):
        n_sample = data_batch[0].shape[0]
        index = np.arange(n_sample)
        np.random.shuffle(index)
        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index, :] if x.ndim > 1 else x[span_index] for x in data_batch]

    def update(self, s_batch, a_batch, r_batch, d_batch, update_ratio):
        advantage_batch, target_value_batch, old_logit_action_probability_batch = [], [], []
        for i in range(len(d_batch)):
            traj_size = len(d_batch[i])
            adv = np.empty(traj_size, dtype=np.float32)

            old_logit, state_value = self.sess.run(
                [self.logit_action_probability, self.state_value],
                feed_dict={self.ob_image: s_batch[i]})

            old_logit_action_probability_batch += old_logit.tolist()[:-1]
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

        old_logit_action_probability_batch = np.asarray(old_logit_action_probability_batch)

        # Train network.
        for _ in tqdm(range(self.training_epoch)):
            # Get training sample generator.
            batch_generator = self._generator(
                [s_batch, a_batch, advantage_batch, old_logit_action_probability_batch, target_value_batch], batch_size=self.training_batchsize)

            while True:
                try:
                    mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_logit_action_probability_batch, mini_target_state_value_batch = next(
                        batch_generator)

                    fd = {
                        self.ob_image: mini_s_batch,
                        self.old_logit_action_probability: mini_old_logit_action_probability_batch,
                        self.action: mini_a_batch,
                        self.advantage: mini_advantage_batch,
                        self.target_state_value: mini_target_state_value_batch,
                        self.moved_lr: self.lr_schedule(update_ratio),
                        self.clip_epsilon: self.clip_schedule(update_ratio)}

                    c_loss, surr, entro, p_ratio, _ = self.sess.run([self.critic_loss,
                                                                     self.surrogate,
                                                                     self.entropy,
                                                                     self.ratio,
                                                                     self.total_train_op],
                                                                    feed_dict=fd)
                except StopIteration:
                    del batch_generator
                    break
        