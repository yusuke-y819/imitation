import gymnasium
from gymnasium import spaces
import numpy as np
from typing import List
from typing import Tuple
from .config import config as conf
from gymnasium.spaces import Box
from PIL import Image
import os
import json

class RacingEnv1(gymnasium.Env):
    ACTION_NAME: List[str] = ['steer', 'throttle']
    VAL_PER_PIXEL: int = 255

    def __init__(self, expert_data):
        # action_spaceを定義
        self.action_space = spaces.Box(
            low=np.array([float(conf['steer_min']), float(conf['throttle_min'])]),
            high=np.array([float(conf['steer_max']), float(conf['throttle_max'])]),
            dtype=np.float32
        )
        
        # observation_spaceを定義
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (3, conf['height'], conf['width']), dtype=np.uint8)

        # expert_dataを受け取る
        # self.expert_data_path = expert_data_path
        # self.expert_data = load_expert_data(self.expert_data_path)
        self.expert_data = expert_data

        self.dirctory = np.random.randint(len(self.expert_data))
        print(self.dirctory)

        self.current_step = 0
        self.total_step = len(self.expert_data[self.dirctory]['images'])

        # rewardの重みを定義
        self.weight = {
            'steer': 1.0,
            'throttle': 1.0
        }

        self.state = None
        self.count = 0
        # self.seed()

    def step(self, action):
        # expert_dataから対応するステップのデータを取得する
        current_expert_data = self.expert_data[self.dirctory]['actions'][self.current_step]

        # rewradを計算する(模倣学習では採用されない)
        # MSEを使用する
        reward = self.calculate_reward(action, current_expert_data)

        # 次のステップに進む
        self.current_step += 1

        # 終了判定
        done = self.current_step == self.total_step - 1

        # truncatedを定義
        truncated = False

        # infoを定義（箱だけ）
        info = {}

        # 次のobservationを定義
        next_observation = self.expert_data[self.dirctory]['images'][self.current_step]

        return next_observation, reward, done, truncated, info, 

    def calculate_reward(self, action, current_expert_data):
        # MSEを使用する
        reward = 0
        for i, action_name in enumerate(self.ACTION_NAME):
            reward += self.weight[action_name] * (action[i] - current_expert_data[i]) ** 2
        
        # normalized_action = np.zeros(2)
        # normalized_expert_action = np.zeros(2)
        # # sterrを正規化
        # normalized_action[0] = (action[0] + 1) / 2
        # normalized_expert_action[0] = (current_expert_data[0] + 1) / 2
        # # throttleを正規化
        # normalized_action[1] = action[1]
        # normalized_expert_action[1] = current_expert_data[1]
        # # MSEを計算
        # reward = np.linalg.norm(normalized_action - normalized_expert_action)
        
        return 1/(reward+1)
    
    def seed(self, seed=None):  # 今のところ使わない
        self.count += 1
        print(self.count)
    
    def reset(self, seed=None, options=None):
        # 環境をリセットする
        self.current_step = 0

        # infoを定義（箱だけ）
        info = {}

        self.dirctory = np.random.randint(len(self.expert_data))
        self.total_step = len(self.expert_data[self.dirctory]['images'])
        print(self.dirctory)
        return self.expert_data[self.dirctory]['images'][self.current_step], info
    
    def expert_data(self):
        return self.expert_data
    