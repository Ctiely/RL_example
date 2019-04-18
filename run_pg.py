#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:25:59 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import logging
    from algorithms.pg import PolicyGradient
    from env.dist_env import BreakoutEnv
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    explore_steps = 1024
    save_model_freq = 100
    
    env = BreakoutEnv(50002, num_envs=20)
    env_ids, states, rewards, dones = env.start()
    pg = PolicyGradient(env.action_space, env.state_space)
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = pg.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        s_batch, a_batch, r_batch, _ = env.get_episodes()
        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        pg.update(s_batch, a_batch, r_batch)
        pg.sw.add_scalar(
                    'epreward_mean',
                    env.mean_reward,
                    global_step=nth_trajectory)

        if nth_trajectory % save_model_freq == 0:
            pg.save_model()
