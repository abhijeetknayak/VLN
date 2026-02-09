import argparse
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import numpy as np
import torch
import quaternion
import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

from internnav.dataset.system2_dataset import System2PixelGoalDataset
from scripts.eval.eval import load_eval_cfg
from easydict import EasyDict as edict



DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5

class PixelGoalEval:
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)

        self.data_args = {"vln_dataset_use": "r2r_125cm_0_30%30",
                          "sample_step": 1, 
                          "num_history": 8,
                          "resize_w": 384,
                          "resize_h": 384 }

        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.local_rank = 0  # only support single gpu eval for now

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        self.dataset = System2PixelGoalDataset(edict(self.data_args))

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        llm_outputs = "↓"

        for idx, data_dict in enumerate(tqdm.tqdm(self.dataset, desc="Evaluating pixel goal dataset")):
            messages = []

            images = data_dict['images']
            gt_pixel_goal = data_dict['action']
            input_img_id = 0

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace(
                '<instruction>.', data_dict['instruction']
            )

            total_images = len(images)
            history_images = total_images - 2

            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * history_images
            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            sources[0]["value"] += f" {prompt}."
            prompt_instruction = copy.deepcopy(sources[0]["value"])
            parts = split_and_clean(prompt_instruction)

            content = []
            for i in range(len(parts)):
                if parts[i] == "<image>":
                    content.append({"type": "image", "image": images[input_img_id]})
                    input_img_id += 1
                else:
                    content.append({"type": "text", "text": parts[i]})

            messages.append({'role': 'user', 'content': content})

            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            messages.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
            )

            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            sources[0]["value"] += f" {prompt}."
            prompt_instruction = copy.deepcopy(sources[0]["value"])
            parts = split_and_clean(prompt_instruction)

            content = []
            for i in range(len(parts)):
                if parts[i] == "<image>":
                    content.append({"type": "image", "image": images[input_img_id]})
                    input_img_id += 1
                else:
                    content.append({"type": "text", "text": parts[i]})

            messages.append({'role': 'user', 'content': content})

            assert input_img_id == len(images), "All images should be used."

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=None,
                    return_dict_in_generate=True,
                ).sequences

            model_output = self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

            if bool(re.search(r'\d', model_output)):  # output pixel goal
                coord = [int(c) for c in re.findall(r'\d+', model_output)]
                pixel_goal = [int(coord[1]), int(coord[0])]

                frame = np.array(images[-1])

                cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=4, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (gt_pixel_goal[0], gt_pixel_goal[1]), radius=8, color=(0, 255, 0), thickness=-1)

                cv2.imwrite(os.path.join(self.output_path, f'{idx:04d}.png'), frame)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)

    args = parser.parse_args()

    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')

    evaluator = PixelGoalEval(evaluator_cfg)

    evaluator._run_eval_system2()