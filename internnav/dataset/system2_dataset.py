import copy
import io
import itertools
import json
import os
import random
import re
import time
import lmdb
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_utils import to_numpy_array
# from easy_dict import EasyDict as edict


import sys
import pdb



lmdb_training = True

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",  # noqa: F541
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}


# R2R_125CM_0_30 = {
#     "data_path": "traj_data/r2r",
#     "height": 125,
#     "pitch_1": 0,
#     "pitch_2": 30,
# }

R2R_125CM_0_30 = {
    "data_path": "/hnvme/workspace/v106be14-nav_data/InternData-N1/vln_ce/lmdbs/r2r",
    # "data_path": "/home/ikep64up/software/InternNav/data",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}



R2R_125CM_0_45 = {
    "data_path": "/hnvme/workspace/v106be14-nav_data/InternData-N1/vln_ce/lmdbs/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

R2R_60CM_15_15 = {
    "data_path": "/mnt/dataset_drive/nav_datasets/raw-datasets/InternData-N1/vln_ce/traj_data/r2r",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

R2R_60CM_30_30 = {
    "data_path": "/mnt/dataset_drive/nav_datasets/raw-datasets/InternData-N1/vln_ce/traj_data/r2r",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

RxR_125CM_0_30 = {
    "data_path": "/mnt/dataset_drive/nav_datasets/raw-datasets/InternData-N1/vln_ce/traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

RxR_125CM_0_45 = {
    "data_path": "traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

RxR_60CM_15_15 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

RxR_60CM_30_30 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_30 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_45 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

SCALEVLN_60CM_30_30 = {
    "data_path": "traj_data/scalevln",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "r2r_125cm_0_30": R2R_125CM_0_30,
    "r2r_125cm_0_45": R2R_125CM_0_45,
    "r2r_60cm_15_15": R2R_60CM_15_15,
    "r2r_60cm_30_30": R2R_60CM_30_30,
    "rxr_125cm_0_30": RxR_125CM_0_30,
    "rxr_125cm_0_45": RxR_125CM_0_45,
    "rxr_60cm_15_15": RxR_60CM_15_15,
    "rxr_60cm_30_30": RxR_60CM_30_30,
    "scalevln_125cm_0_30": SCALEVLN_125CM_0_30,
    "scalevln_125cm_0_45": SCALEVLN_125CM_0_45,
    "scalevln_60cm_30_30": SCALEVLN_60CM_30_30,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
TRAJ_TOKEN_INDEX = 151670
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_TRAJ_TOKEN = "<traj>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def get_annotations_from_lerobot_data(data_path, setting):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow.parquet as pq

    annotations = {
        "axis_align_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "episodes": [],
    }
    scene_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    def process_scene(scene_id):
        scene_path = os.path.join(data_path, scene_id)
        episodes = read_jsonl(os.path.join(scene_path, "meta", "episodes.jsonl"))
        scene_annotations = []

        for ep in episodes:
            ep_id = ep["episode_index"]
            ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
            ep_len = ep["length"]
            parquet_path = os.path.join(
                scene_path, "data", f"chunk-{ep_id // 1000:03d}", f"episode_{ep_id:06d}.parquet"
            )

            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            ep_actions = df["action"].tolist()

            pose_key = f"pose.{setting}"
            goal_key = f"goal.{setting}"
            relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"

            if pose_key in df.columns and goal_key in df.columns and relative_goal_frame_id_key in df.columns:
                ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
                ep_pixel_goals = [
                    [df[relative_goal_frame_id_key][idx].tolist(), df[goal_key][idx].tolist()] for idx in range(len(df))
                ]
            else:
                print(f"Warning: Missing data for setting {setting} in episode {ep_id}, filling with defaults.")

            assert len(ep_actions) == ep_len, f"Action length mismatch in episode {ep_id}"

            for ep_instruction in ep_instructions:
                episode = {
                    "id": ep_id,
                    "video": f"{data_path}/{scene_id}/videos/chunk-{ep_id // 1000:03d}",
                    "instructions": ep_instruction,
                    "actions": ep_actions,
                    "length": ep_len,
                    f"poses_{setting}": ep_poses,
                    "pixel_goals": ep_pixel_goals,
                }
                scene_annotations.append(episode)

        return scene_annotations

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_scene, scene_id): scene_id for scene_id in scene_ids}
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                scene_annotations = future.result()
                annotations["episodes"].extend(scene_annotations)
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")

    return annotations

def get_annotations_from_lmdb(data_path, setting):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow as pa
    import pyarrow.parquet as pq

    annotations = {
        "axis_align_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "episodes": [],
    }

    scene_episodes = []

    # Open lmdb to extract episode keys
    env = lmdb.open(data_path, subdir=False, readonly=True, lock=False)

    with env.begin() as txn:
        keys = [key.decode('utf-8') for key, _ in txn.cursor()]
        for key in keys:
            if key.endswith('episodes.jsonl'):
                scene_episodes.append(key)

    def process_scene(data_path, lmdb_key):
        scene_annotations = []
        env = lmdb.open(data_path, subdir=False, readonly=True, lock=False)

        with env.begin() as txn:
            episodes_data = txn.get(lmdb_key.encode('utf-8'))
            scene_tar_gz = lmdb_key.split('/')[0] # Lmdb starts with scene/...
            scene_name = scene_tar_gz.replace('.tar.gz', '')

            episodes = [json.loads(line) for line in episodes_data.decode('utf-8').splitlines()]
            for ep in episodes:
                ep_id = ep["episode_index"]
                ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
                ep_len = ep["length"]
                parquet_key = f"{scene_name}/data/chunk-{ep_id // 1000:03d}/episode_{ep_id:06d}.parquet"
                parquet_data = txn.get(parquet_key.encode('utf-8'))
                
                parquet_data = pa.py_buffer(parquet_data)
                table = pq.read_table(pa.BufferReader(parquet_data))

                df = table.to_pandas()

                ep_actions = df["action"].tolist()

                pose_key = f"pose.{setting}"
                goal_key = f"goal.{setting}"
                relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"

                if pose_key in df.columns and goal_key in df.columns and relative_goal_frame_id_key in df.columns:
                    ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
                    ep_pixel_goals = [
                        [df[relative_goal_frame_id_key][idx].tolist(), df[goal_key][idx].tolist()] for idx in range(len(df))
                    ]
                else:
                    print(f"Warning: Missing data for setting {setting} in episode {ep_id}, filling with defaults.")

                assert len(ep_actions) == ep_len, f"Action length mismatch in episode {ep_id}"

                for ep_instruction in ep_instructions:
                    episode = {
                        "id": ep_id,
                        "video": f"{scene_name}/videos/chunk-{ep_id // 1000:03d}",
                        "instructions": ep_instruction,
                        "actions": ep_actions,
                        "length": ep_len,
                        f"poses_{setting}": ep_poses,
                        "pixel_goals": ep_pixel_goals,
                    }
                    scene_annotations.append(episode)
        return scene_annotations        

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_scene, data_path, lmdb_key): lmdb_key for lmdb_key in scene_episodes}
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                scene_annotations = future.result()
                annotations["episodes"].extend(scene_annotations)
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")

    return annotations


class System2PixelGoalDataset(Dataset):
    def __init__(self, data_args):
        super(System2PixelGoalDataset, self).__init__()
        dataset = data_args.vln_dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")

        self.sample_step = data_args.sample_step
        self.list_data_dict = []

        for data in dataset_list:
            sampling_rate = data.get("sampling_rate", 1.0)
            height = data.get("height", None)
            pitch_1 = data.get("pitch_1", None)
            pitch_2 = data.get("pitch_2", None)

            data_path = data['data_path']
            setting = f'{height}cm_{pitch_2}deg'
            lmdb_path = os.path.join(data_path, 'metadata_parquet.lmdb')

            if lmdb_training:
                annotations = get_annotations_from_lmdb(lmdb_path, setting)
            else:
                annotations = get_annotations_from_lerobot_data(data_path, setting)

            pixel_goal_list = []
            list_data_dict = []

            for item in annotations['episodes']:
                ep_id = item['id']
                instruction = item['instructions']
                video = item['video']
                actions = item['actions'][1:] + [0]
                pixel_goals = item['pixel_goals']
                poses = item[f'poses_{height}cm_{pitch_2}deg']

                actions_len = len(actions)
                if actions_len < 4:
                    continue

                num_rounds = actions_len // self.sample_step
                for n in range(num_rounds + 1):
                    if n * self.sample_step == actions_len or n * self.sample_step == actions_len - 1:
                        continue
                    start_frame_id = n * self.sample_step
                    action_flag = actions[start_frame_id]
                    pixel_goal = pixel_goals[start_frame_id]
                    if pixel_goal[0] == -1:
                        if action_flag == 1:
                            continue
                    else:
                        goal_len = pixel_goal[0]
                        if goal_len < 3:
                            continue
                        action = pixel_goal[1]
                        pose = poses[start_frame_id : start_frame_id + goal_len + 1]
                        pixel_goal_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                (start_frame_id, start_frame_id + goal_len + 1),
                                action,
                                pose,
                            )
                        )

            list_data_dict = pixel_goal_list
            rank0_print(len(pixel_goal_list))
            if sampling_rate < 1.0:
                list_data_dict = random.sample(list_data_dict, int(len(list_data_dict) * sampling_rate))
                print(f"sampling {len(list_data_dict)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            self.list_data_dict.extend(list_data_dict)

        self.num_history = data_args.num_history
        self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        if lmdb_training:
            return self.__getitem_lmdb__(i)
        else:
            return self.__getitem_lerobot__(i)
        

    def __getitem_lmdb__(self, i):
        (
            ep_id,
            data_path,
            video,
            height,
            pitch_1,
            pitch_2,
            instruction,
            (start_frame_id, end_frame_id),
            action,
            pose,
        ) = self.list_data_dict[i]
        if start_frame_id != 0:
            history_id = np.unique(np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32)).tolist()
        else:
            history_id = []

        data_dict = {}
        images = []

        obs_lmdb_path = os.path.join(data_path, 'observations.lmdb')
        env = lmdb.open(obs_lmdb_path, subdir=False, readonly=True, lock=False)

        with env.begin() as txn:
            for id in range(0, end_frame_id):
                image_file = os.path.join(video, f"observation.images.rgb.{height}cm_{pitch_1}deg", f"episode_{ep_id:06d}_{id}.jpg")

                # Get image from lmdb
                image_data = txn.get(image_file.encode('utf-8'))
                image = Image.open(io.BytesIO(image_data)).convert('RGB')

                lookdown_image_key = image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')
                lookdown_image_data = txn.get(lookdown_image_key.encode('utf-8'))
                lookdown_image = Image.open(io.BytesIO(lookdown_image_data)).convert('RGB')

                if id in history_id or id == start_frame_id:
                    image = image.resize((self.data_args.resize_w, self.data_args.resize_h))
                    images.append(image)
                    if id == start_frame_id and pose is not None:
                        images.append(lookdown_image)
 

        data_dict["images"] = images
        data_dict["action"] = action
        data_dict["instruction"] = instruction

        return data_dict

    def __getitem_lerobot__(self, i):
        (
            ep_id,
            data_path,
            video,
            height,
            pitch_1,
            pitch_2,
            instruction,
            (start_frame_id, end_frame_id),
            action,
            pose,
        ) = self.list_data_dict[i]
        if start_frame_id != 0:
            history_id = np.unique(np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32)).tolist()
        else:
            history_id = []

        data_dict = {}
        images = []

        for id in range(0, end_frame_id):
            image_file = os.path.join(
                video, f"observation.images.rgb.{height}cm_{pitch_1}deg", f"episode_{ep_id:06d}_{id}.jpg"
            )
            image = Image.open(image_file).convert('RGB')
            lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')).convert('RGB')

            if id in history_id or id == start_frame_id:
                image = image.resize((self.data_args['resize_w'], self.data_args['resize_h']))
                images.append(image)
                if id == start_frame_id and pose is not None:
                    images.append(lookdown_image)
        
        data_dict["images"] = images
        data_dict["action"] = action
        data_dict["instruction"] = instruction
        
        return data_dict



if __name__ == "__main__":
    pass
    # data_args = {
    #     "vln_dataset_use": "r2r_125cm_0_30%30",
    #     "sample_step": 1,
    #     "num_history": 8,
    #     "resize_w": 384,    
    #     "resize_h": 384,
    # }
    # dataset = System2PixelGoalDataset(data_args=edict(data_args))

    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     print(i, data['images'][-1].size)    
