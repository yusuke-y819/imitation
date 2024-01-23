import gymnasium as gym
from gym import spaces
import numpy as np
from typing import List
from typing import Tuple
from .config import config as conf
from gym.spaces import Box
from PIL import Image
import os
import json

class MyEnv(gym.Env):
    ACTION_NAME: List[str] = ['steer', 'throttle']
    VAL_PER_PIXEL: int = 255

    def __init__(self, expert_data):
        # action_spaceを定義
        self.action_space = spaces.Box(
            low=np.array([-float(conf['steer_min']), float(conf['throttle_min'])]),
            high=np.array([float(conf['steer_max']), float(conf['throttle_max'])]),
            dtype=np.float32
        )
        
        # observation_spaceを定義
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (conf['height'], conf['width'], 3), dtype=np.uint8)

        # expert_dataを受け取る
        # self.expert_data_path = expert_data_path
        # self.expert_data = load_expert_data(self.expert_data_path)
        self.expert_data = expert_data

        self.current_step = 0
        self.total_step = len(self.expert_data['images'])

        # rewardの重みを定義
        self.weight = {
            'steer': 1.0,
            'throttle': 1.0
        }

        self.state = None
        self.seed()

    def _load_expert_data(self):
        # expert_dataをロード
        self.expert_data = load_expert_data(self.expert_data_path)
        self.total_step = len(self.expert_data['images'])

    def step(self, action):
        # expert_dataから対応するステップのデータを取得する
        current_expert_data = self.expert_data[self.current_step]

        # rewradを計算する(模倣学習では採用されない)
        # MSEを使用する
        reward = self.calculate_reward(action, current_expert_data)

        # 次のステップに進む
        self.current_step += 1

        # 終了判定
        done = self.current_step == self.total_step

        # infoを定義（箱だけ）
        info = {}

        # 次のobservationを定義
        next_observation = self.expert_data['images'][self.current_step]

        return next_observation, reward, done, info

    def calculate_reward(self, action, current_expert_data):
        # MSEを使用する
        reward = 0
        for i, action_name in enumerate(self.ACTION_NAME):
            reward += self.weight[action_name] * (action[i] - current_expert_data[action_name]) ** 2
        return reward
    
    def seed(self, seed=None):  # 今のところ使わない
        self.seed(seed)
    
    def reset(self):
        # 環境をリセットする
        self.current_step = 0
        return self.expert_data['images'][self.current_step]
    
def load_expert_data(data_path):
    # データの読み込み
    meta_path = os.path.join(data_path, 'meta.json')
    with open(meta_path, 'r') as meta_file:
        meta_data = json.load(meta_file)

    # 画像データのファイル名リストを取得
    # '.jpg'で終わるファイルのみを取得する
    image_file_list = [record_file for record_file in os.listdir(data_path) if record_file.endswith('.jpg')]

    # expert_dataの初期化
    expert_data = {'images': [], 'actions': []}

    for image_file in image_file_list:
        # レコードのファイルパスを構築
        record_file = image_file.replace('image_array', 'record').replace('.jpg', '.json')
        record_path = os.path.join(data_path, record_file)

        # レコードの読み込み
        with open(record_path, 'r') as record_file:
            record_data = json.load(record_file)

        # 画像データの読み込み
        image_path = os.path.join(data_path, image_file)
        image_data = np.array(Image.open(image_path))

        # expert_dataに追加
        expert_data['images'].append(image_data)
        expert_data['actions'].append([record_data['user/angle'], record_data['user/throttle']])

    return expert_data
# import gymnasium as gym
# from gym import spaces
# import numpy as np
# from typing import List
# from typing import Tuple
# from .config import config as conf
# from gym.spaces import Box
# from PIL import Image
# import os
# import json



# class MyEnv(gym.Env):
#     # metadata = {'render.modes': ['human', "rgb_array"]}
#     ACTION_NAME: List[str] = ['steer', 'throttle']
#     VAL_PER_PIXEL: int = 255

#     def __init__(self, expert_data):
#         # action_spaceを定義
#         self.action_space = spaces.Box(
#             low=np.array([-float(conf['steer_min']), float(conf['throttle_min'])]),
#             high=np.array([float(conf['steer_max']), float(conf['throttle_max'])]),
#             dtype=np.float32
#             )
        
#         # observation_spaceを定義
#         self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (conf['height'], conf['width'], 3), dtype=np.uint8)

#         # expert_dataを受け取る
#         self.expert_data = load_expert_data(expert_data)

#         self.current_step = 0
#         self.total_step = len(expert_data['images'])

#         # rewardの重みを定義
#         self.weight = {
#             'steer': 1.0,
#             'throttle': 1.0
#         }

#         self.state = None
#         self.seed()


#     def step(self, action):
#         # expert_dataから対応するステップのデータを取得する
#         current_expert_data = self.expert_data[self.current_step]

#         # rewradを計算する(模倣学習では採用されない)
#         # MSEを使用する
#         reward = self.calculate_reward(action, current_expert_data)

#         # 次のステップに進む
#         self.current_step += 1

#         # 終了判定
#         done = self.current_step == self.total_step

#         # infoを定義（箱だけ）
#         info = {}

#         # 次のobservationを定義
#         next_observation = self.expert_data['images'][self.current_step]

#         return next_observation, reward, done, info

#     def calculate_reward(self, action, current_expert_data):
#         # MSEを使用する
#         reward = 0
#         for i, action_name in enumerate(self.ACTION_NAME):
#             reward += self.weight[action_name] * (action[i] - current_expert_data[action_name]) ** 2
#         return reward
    
#     def seed(self, seed=None):  # 今のところ使わない
#         self.seed(seed)
    
#     def reset(self):
#         # 環境をリセットする
#         self.current_step = 0
#         return self.expert_data['images'][self.current_step]
    
# def load_expert_data(data_path):
#     # データの読み込み
#     meta_path = os.path.join(data_path, 'meta.json')
#     with open(meta_path, 'r') as meta_file:
#         meta_data = json.load(meta_file)

#     # 画像データのファイル名リストを取得
#     # '.jpg'で終わるファイルのみを取得する
#     image_file_list = [record_file for record_file in os.listdir(data_path) if record_file.endswith('.jpg')]

#     # expert_dataの初期化
#     expert_data = {'images': [], 'actions': []}

#     for image_file in image_file_list:
#         # レコードのファイルパスを構築
#         record_file = image_file.replace('image_array', 'record').replace('.jpg', '.json')
#         record_path = os.path.join(data_path, record_file)

#         # レコードの読み込み
#         with open(record_path, 'r') as record_file:
#             record_data = json.load(record_file)

#         # 画像データの読み込み
#         image_path = os.path.join(data_path, image_file)
#         image_data = np.array(Image.open(image_path))

#         # expert_dataに追加
#         expert_data['images'].append(image_data)
#         expert_data['actions'].append([record_data['user/angle'], record_data['user/throttle']])

#     # # 特定のディレクトリだけを選択
#     # target_directories = ['tub_9_24-01-09', 'tub_11_24-01-09']

#     # for directory in target_directories:
#     #     directory_path = os.path.join(data_path, directory)

#     #     for image_file in [record_file for record_file in os.listdir(directory_path) if record_file.endswith('.jpg')]:
#     #         # レコードのファイルパスを構築
#     #         record_file = image_file.replace('image_array', 'record').replace('.jpg', '.json')
#     #         record_path = os.path.join(directory_path, record_file)

#     #         # レコードの読み込み
#     #         with open(record_path, 'r') as record_file:
#     #             record_data = json.load(record_file)

#     #         # 画像データの読み込み
#     #         image_path = os.path.join(directory_path, image_file)
#     #         image_data = np.array(Image.open(image_path))

#     #         # expert_dataに追加
#     #         expert_data['images'].append(image_data)
#     #         expert_data['actions'].append([record_data['user/angle'], record_data['user/throttle']])

#     return expert_data

# # # データのパス
# # data_path = 'path/to/data/tub_9_24-01-09'

# # # エキスパートデータの読み込み
# # expert_data = load_expert_data(data_path)

# # # 環境の初期化
# # env = MyEnv(expert_data)

# # num_episodes = 1000

# # # 学習ループなどで利用
# # for _ in range(num_episodes):
# #     action = policy.predict(observation)  # ポリシーによるアクション予測
# #     observation, reward, done, info = env.step(action)
# #     if done:
# #         observation = env.reset()