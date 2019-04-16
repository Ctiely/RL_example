#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:18:44 2019

@author: clytie
"""

import sys
import random
import signal
import numpy as np
from collections import deque
from dist_rl.wrapper import EnvironmentWrapper
from env.envs import GymEnv

def signal_handler(sig, frame):
    """中止程序"""
    print("catch Ctrl-c, exit")
    sys.exit(0)


class BreakoutEnv(object):
    def __init__(self, port, index=0, seed=42, frame_stack=4, 
                 episode_life=False, clip_rewards=True):
        self.env = GymEnv("BreakoutNoFrameskip-v4", 
                          index, seed, 
                          episode_life=episode_life, 
                          clip_rewards=clip_rewards)
        self._port = port
        self.frame_stack = frame_stack
        self.env_wrapper = EnvironmentWrapper(("localhost", port))
        self.st = np.zeros((84, 84, self.frame_stack), dtype=np.uint8)
        self.frames = deque([], maxlen=self.frame_stack)

    def start(self):
        state = self.env.reset()
        while len(self.frames) < self.frame_stack:
            self.frames.append(state)
        
        reward = 0.0
        done = False
        while True:
            for idx, each in enumerate(self.frames):
                self.st[:, :, idx: idx + 1] = each
            self.env_wrapper.put_srd(self.st, reward, done)
            action = self.env_wrapper.get_a()
            if done:
                state = self.env.reset()
                for _ in range(self.frame_stack):
                    self.frames.append(state)
                reward = 0.0
                done = False
            else:
                state, reward, done, _ = self.env.step(action)
                self.frames.append(state)
    
signal.signal(signal.SIGINT, signal_handler)                
     
           
if __name__ == "__main__":
    port = sys.argv[1]
    index = random.randint(0, 100)
    # print("index: ", index)
    env = BreakoutEnv(int(port), index)
    env.start()
                
