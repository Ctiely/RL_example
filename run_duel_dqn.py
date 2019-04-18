#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:38:15 2019

@author: clytie
"""

if __name__ == "__main__":
    import logging
    import numpy as np
    from tqdm import tqdm
    from env.dist_env import BreakoutEnv
    from algorithms.dqn import ReplayBuffer
    from algorithms.duel_dqn import DuelDQN
    

    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    memory = ReplayBuffer(max_size=200000)
    env = BreakoutEnv(50001, num_envs=20)
    env_ids, states, rewards, dones = env.start()
    print("pre-train: ")
    for _ in tqdm(range(5000)):
        env_ids, states, rewards, dones = env.step(env_ids, np.random.randint(env.action_space, size=env.num_srd))
    trajs = env.get_episodes()

    memory.add(trajs)
    DuelDQNetwork = DuelDQN(env.action_space, env.state_space, save_path="./duel_dqn_log")
    
    print("start train: ")
    for step in range(10000000):
        for _ in range(20):
            actions = DuelDQNetwork.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)
        if step % 100 == 0:
            logging.info(f'>>>>{env.mean_reward}, nth_step{step}, buffer{len(memory)}')
        trajs = env.get_episodes()
        memory.add(trajs)
        batch_samples = memory.sample(256)
        DuelDQNetwork.update(batch_samples, sw_dir="duel_dqn")

    env.close()