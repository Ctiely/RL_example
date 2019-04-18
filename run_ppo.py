#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:58:05 2019

@author: clytie
"""

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import logging
    from algorithms.ppo import PPO
    from env.dist_env import BreakoutEnv
    
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s\t%(filename)s\t%(lineno)d\t%(message)s')

    explore_steps = 512
    total_updates = 500
    save_model_freq = 30
    
    env = BreakoutEnv(50002, num_envs=20)
    env_ids, states, rewards, dones = env.start()
    ppo = PPO(env.action_space, env.state_space)
    
    nth_trajectory = 0
    while nth_trajectory < total_updates:
        nth_trajectory += 1
        for _ in tqdm(range(explore_steps)):
            actions = ppo.get_action(np.asarray(states))
            env_ids, states, rewards, dones = env.step(env_ids, actions)

        s_batch, a_batch, r_batch, d_batch = env.get_episodes()
        logging.info(
            f'>>>>{env.mean_reward}, nth_trajectory{nth_trajectory}')
        
        ppo.update(s_batch, a_batch, r_batch, d_batch,
                       min(0.9, nth_trajectory / total_updates))
        
        if nth_trajectory % save_model_freq == 0:
            ppo.save_model()

