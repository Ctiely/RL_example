#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:17:55 2019

@author: clytie
"""

if __name__ == "__main__":
    import gym
    import random
    from env.envs import GymEnv
    from env.dist_env import BreakoutEnv


    env = gym.make("BreakoutNoFrameskip-v4")
    print("图像尺寸:", env.observation_space.shape)
    print("动作空间:", env.action_space.n)

    state = env.reset()
    env.render()
    done = False
    while not done:
        state, reward, done, _ = env.step(random.randint(0, env.action_space.n - 1))
        env.render()


    env = GymEnv("BreakoutNoFrameskip-v4", 0, 42, episode_life=False)
    action_space = env.get_action_space().n
    state = env.reset()
    env.render()
    done = False
    while not done:
        state, reward, done, _ = env.step(random.randint(0, action_space - 1))
        env.render()


    dist_env = BreakoutEnv(port=49999)
    env_ids, states, rewards, dones = dist_env.start()
    for _ in range(100):
        env_ids, states, rewards, dones = dist_env.step(env_ids, [2, 3])

    trajs = dist_env.get_episodes()
