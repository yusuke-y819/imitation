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

class RacingEnv2(gymnasium.Env):
    ACTION_NAME: List[str] = ['steer', 'throttle']
    VAL_PER_PIXEL: int = 255

    def __init__(self):
        # action_spaceを定義
        self.action_space = spaces.Box(
            low=np.array([float(conf['steer_min']), float(conf['throttle_min'])]),
            high=np.array([float(conf['steer_max']), float(conf['throttle_max'])]),
            dtype=np.float32
        )
        
        # observation_spaceを定義
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (conf['height'], conf['width'], 3), dtype=np.uint8)

        # expert_dataを受け取る
        # self.expert_data_path = expert_data_path
        # self.expert_data = load_expert_data(self.expert_data_path)
        self.expert_data = load_expert_data('../../autorace/data/tub_9_24-01-09')

        self.current_step = 0
        self.total_step = len(self.expert_data['images'])

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
        current_expert_data = self.expert_data['actions'][self.current_step]

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
        next_observation = self.expert_data['images'][self.current_step]

        return next_observation, reward, done, truncated, info, 

    def calculate_reward(self, action, current_expert_data):
        # MSEを使用する
        reward = 0
        for i, action_name in enumerate(self.ACTION_NAME):
            reward += self.weight[action_name] * (action[i] - current_expert_data[i]) ** 2
        return 1/(reward+1)
    
    def seed(self, seed=None):  # 今のところ使わない
        self.count += 1
        print(self.count)
    
    def reset(self, seed=None, options=None):
        # 環境をリセットする
        self.current_step = 0

        # infoを定義（箱だけ）
        info = {}
        return self.expert_data['images'][self.current_step], info
    
    def expert_data(self):
        return self.expert_data
    

def load_expert_data(data_path):
    # expert_dataの初期化
    expert_data = {'images': [], 'actions': []}
    # print(expert_data)
    # count = 0

    # 画像ファイルに対応するJSONファイルを取得
    json_file_list = [json_file for json_file in os.listdir(data_path) if json_file.startswith('record_') and json_file.endswith('.json')]
    # print(len(json_file_list))

    for json_file in json_file_list:
        # count += 1
        # print(count)
        # print(json_file)

        # レコードのファイルパスを構築
        json_path = os.path.join(data_path, json_file)
        # print(json_path)

        # レコードの読み込み
        try:
            with open(json_path, 'r') as json_file:
                record_data = json.load(json_file)
        except FileNotFoundError:
            print(f"エラー：{json_path} でJSONファイルが見つかりませんでした。")
            continue
        except json.JSONDecodeError:
            print(f"エラー：{json_path} のJSONファイルのデコードに失敗しました。")
            continue

        # 画像データの読み込み
        image_file = record_data.get('cam/image_array', '')  # 画像ファイル名をJSONから取得
        # print(image_file)
        image_path = os.path.join(data_path, image_file)
        # print(image_path)
        try:
            image_data = np.array(Image.open(image_path))
        except FileNotFoundError:
            print(f"エラー：{image_path} で画像ファイルが見つかりませんでした。")
            continue

        # expert_dataに追加
        expert_data['images'].append(image_data)
        expert_data['actions'].append([record_data.get('user/angle', 0), record_data.get('user/throttle', 0)])

    return expert_data
