#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:25:59 2019

@author: clytie
"""

from collections import deque


def collect_complete_traj(pg, env):
    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    frames = deque([state] * 4, maxlen=4)
    st = np.zeros((84, 84, 4), dtype=np.uint8)
    total_reward = 0
    while True:
        for idx, each in enumerate(frames):
            st[:, :, idx: idx + 1] = each
        states.append(st.copy()) # must be value
        
        action = pg.get_action(np.asarray([st]))[0]
        state, reward, done, _ = env.step(action)
        
        frames.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        total_reward += reward
        if done:
            for idx, each in enumerate(frames):
                st[:, :, idx: idx + 1] = each
            states.append(st.copy()) # must be value
            break
    return states, actions, rewards, dones, total_reward

if __name__ == "__main__":
    import numpy as np
    import random
    import logging
    from tqdm import tqdm
    from algorithms.pg import PolicyGradient
    from env.envs import GymEnv
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

    save_model_freq = 100
    num_traj = 20
    action_space = 4
    state_space = (84, 84, 4)
    
    env = GymEnv("BreakoutNoFrameskip-v4", random.randint(0, 100), 42)
    pg = PolicyGradient(action_space, state_space, save_path="./pg_log")
    
    total_reward = deque([], maxlen=500)
    nth_trajectory = 0
    while True:
        nth_trajectory += 1
        
        s_batch, a_batch, r_batch = [], [], []
        for _ in tqdm(range(num_traj)):    
            s_traj, a_traj, r_traj, _ , reward= collect_complete_traj(pg, env)
            s_batch.append(s_traj)
            a_batch.append(a_traj)
            r_batch.append(r_traj)
            total_reward.append(reward)
        
        mean_reward = np.mean(total_reward)
        logging.info(
            f'>>>>{np.round(mean_reward, 5)}, nth_trajectory{nth_trajectory}')
        
        pg.update(s_batch, a_batch, r_batch)
        pg.sw.add_scalar(
                    'epreward_mean',
                    mean_reward,
                    global_step=nth_trajectory)

        if nth_trajectory % save_model_freq == 0:
            pg.save_model()
            num_traj -= 1
            num_traj = max(5, num_traj)
