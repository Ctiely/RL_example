#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:06:40 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    from env.dist_env import BreakoutEnv
    from algorithms.dqn import ReplayBuffer, DQN

    memory = ReplayBuffer()
    env = BreakoutEnv(49999, num_envs=3)
    env_ids, states, rewards, dones = env.start()
    for _ in range(100):
        env_ids, states, rewards, dones = env.step(env_ids, np.random.randint(env.action_space, size=env.num_srd))
    trajs = env.get_episodes()

    memory.add(trajs)
    DQNetwork = DQN(env.action_space, env.state_space)

    for step in tqdm(range(100)):
        for _ in range(100):
            actions = DQNetwork.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        trajs = env.get_episodes()
        memory.add(trajs)
        batch_samples = memory.sample(128)
        DQNetwork.update(batch_samples)

    env.close()