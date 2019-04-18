#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:06:40 2019

@author: clytie
"""

if __name__ == "__main__":
    import logging
    import numpy as np
    from tqdm import tqdm
    from env.dist_env import BreakoutEnv
    from algorithms.dqn import ReplayBuffer, DQN


    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    memory = ReplayBuffer(max_size=500000)
    env = BreakoutEnv(49999, num_envs=20)
    env_ids, states, rewards, dones = env.start()
    print("pre-train: ")
    for _ in tqdm(range(5000)):
        env_ids, states, rewards, dones = env.step(env_ids, np.random.randint(env.action_space, size=env.num_srd))
    trajs = env.get_episodes()

    memory.add(trajs)
    DQNetwork = DQN(env.action_space, env.state_space, save_path="./dqn_log")
    
    print("start train: ")
    for step in range(10000000):
        for _ in range(20):
            actions = DQNetwork.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        if step % 10 == 0:
            logging.info(f'>>>>{env.mean_reward}, nth_step{step}, buffer{len(memory)}')
        trajs = env.get_episodes()
        memory.add(trajs)
        for _ in range(10):
            batch_samples = memory.sample(32)
            DQNetwork.update(batch_samples, sw_dir="dqn")

    env.close()