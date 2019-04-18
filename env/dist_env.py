#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:27:29 2019

@author: clytie
"""

import os
import time
import subprocess
from dist_rl.wrapper import AgentWrapper


env_path = os.path.dirname(__file__)
cur_path = os.getcwd()


class BaseEnv(object):
    def __init__(self, name, port=None, mode="train"):
        self.port = port
        self.game = name
        self.mode = mode
        self.agent = AgentWrapper(self.port)
        self.agent.setDaemon(True)
        self.agent.start()
        self.processes = []

    def _start_env(self, num_envs):
        os.chdir(f"{env_path}")
        worker_cmd = f'''python {env_path}/{self.game}.py {self.port} {self.mode}'''
        for _ in range(num_envs):
            self.processes.append(subprocess.Popen(worker_cmd, shell=True))
        os.chdir(cur_path)

    def _get_srd(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def step(self, env_ids, actions):
        raise NotImplementedError

    def close(self):
        if self.processes:
            for process in self.processes:
                process.kill()
            time.sleep(0.1)
            self.processes = []

    def get_episodes(self, return_total_traj=False):
        raise NotImplementedError

class BreakoutEnv(BaseEnv):
    def __init__(self, port, num_envs=4, mode="train"):
        super().__init__("breakout", port, mode)
        self.num_envs = num_envs
        self.num_srd = max(1, self.num_envs // 2)
    
    @property
    def state_space(self):
        return (84, 84, 4)
    
    @property
    def action_space(self):
        return 4

    def _get_srd(self):
        env_ids, states, rewards, dones = self.agent.get_srd_batch(self.num_srd)
        return env_ids, states, rewards, dones

    def start(self):
        if len(self.processes) == 0:
            self._start_env(self.num_envs)
        return self._get_srd()

    def step(self, env_ids, actions):
        assert len(actions) == self.num_srd
        self.agent.put_a_batch(env_ids, actions)
        return self._get_srd()

    def get_episodes(self, return_total_traj=False):
        withdraw_running = False if return_total_traj else True
        return self.agent.get_episodes(withdraw_running=withdraw_running)

    @property
    def mean_reward(self):
        return self.agent.statistics()["mean_reward"]