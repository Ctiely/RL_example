#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:55:03 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    import time
    from tqdm import tqdm
    from env.dist_env import BreakoutEnv
    from algorithms.dqn import DQN


    DQNetwork = DQN(4, (84, 84, 4), epsilon_schedule=lambda x: 0, save_path="./dqn_log")
    env = BreakoutEnv(4999, num_envs=1, mode="test")
    env_ids, states, _, _ = env.start()
    for _ in tqdm(range(10000)):
        time.sleep(0.1)
        actions = DQNetwork.get_action(np.asarray(states))
        env_ids, states, _, _ = env.step(env_ids, actions)
    env.close()
