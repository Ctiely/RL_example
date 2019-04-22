#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:59:46 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    import time
    from tqdm import tqdm
    from env.dist_env import BreakoutEnv
    from algorithms.pg import PolicyGradient


    pg = PolicyGradient(4, (84, 84, 4), temperature=0.1, save_path="./pg_log")
    env = BreakoutEnv(4999, num_envs=1, mode="test")
    env_ids, states, _, _ = env.start()
    for _ in tqdm(range(10000)):
        time.sleep(0.1)
        actions = pg.get_action(np.asarray(states))
        env_ids, states, _, _ = env.step(env_ids, actions)
    env.close()
