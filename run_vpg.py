#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:49:34 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import logging
    from algorithms.vpg import VanillaPolicyGradient
    from env.dist_env import BreakoutEnv
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    explore_steps = 512
    total_updates = 2000
    save_model_freq = 100
    
    env = BreakoutEnv(50001, num_envs=20)
    env_ids, states, rewards, dones = env.start()
    vpg = VanillaPolicyGradient(env.action_space, env.state_space)
    
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = vpg.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        s_batch, a_batch, r_batch, d_batch = env.get_episodes()
        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        vpg.update(s_batch, a_batch, r_batch, d_batch)
        vpg.sw.add_scalar(
                    'epreward_mean',
                    env.mean_reward,
                    global_step=nth_trajectory)

        if nth_trajectory % save_model_freq == 0:
            vpg.save_model()
    env.close()
