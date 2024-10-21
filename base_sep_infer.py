import torch
import numpy as np
import os
import argparse
from typing import Tuple, List
from PIL import Image
import copy
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
import json

try:
    from inference.models import YOLOWorld
    from src.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from src.efficientvit.sam_model_zoo import create_sam_model
    import supervision as sv
except:
    print("YoloWorld can not be load")

try:
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, predict
    from segment_anything import build_sam, SamPredictor
    import groundingdino.datasets.transforms as T
except:
    print("groundingdino can not be load")

import cv2
import math
from diffusers.utils import load_image

from utils.baseline_sep_utils import sample_image, build_model_sd, build_dino_segment_model, predict_mask, \
    draw_kps_multi


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    # general param
    parser.add_argument('--seed', default=53, type=int)
    # save path
    parser.add_argument('--save_dir', default='results/base_infer_result', type=str)

    # json path
    parser.add_argument('--benchmark_file_path', default='./benchmark/benchmark.json')

    # instantID
    parser.add_argument('--pretrained_model', default='./checkpoint/YamerMIX_v8', type=str)
    parser.add_argument('--controlnet_path', default='./checkpoint/InstantID/ControlNetModel', type=str)
    parser.add_argument('--face_adapter_path', default='./checkpoint/InstantID/ip-adapter.bin', type=str)
    parser.add_argument('--antelopev2_path', default='./checkpoint/antelopev2', type=str)

    # instant ID param
    parser.add_argument('--cfg_scale', default=3.0, type=float)
    parser.add_argument('--IdentityNet_rate', default=0.8, type=float)
    parser.add_argument('--adapter_ratio', default=0.8, type=float)
    parser.add_argument('--controlNet_ratio', default=0.8, type=float)

    # other control
    parser.add_argument('--spatial_condition', default='', type=str)
    parser.add_argument('--t2i_controlnet_path', default='', type=str)
    parser.add_argument('--style_lora', default='', type=str)

    # SEG
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # make base folder
    save_dir = args.save_dir
    benchmark_file_path = args.benchmark_file_path
    os.makedirs(save_dir, exist_ok=True)

    # base param
    width, height = 1024, 1024
    kwargs = {
        'height': height,
        'width': width,
        't2i_image': None,
        't2i_controlnet_conditioning_scale': args.controlNet_ratio,
    }
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    prompts_tmp = ["aaa"] * 2
    # load model
    pipe, controller, pipe_concepts, face_app = build_model_sd(args.pretrained_model, args.controlnet_path,
                                                               args.face_adapter_path, device, prompts_tmp,
                                                               args.antelopev2_path, width // 32, height // 32,
                                                               args.style_lora, args.t2i_controlnet_path,
                                                               args.adapter_ratio)

    detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)

    # setting
    pos_prefix = "masterpiece, close-up portrait photo, front large face photo"
    pos_suffix = "photographic"
    neg_prefix = "back view, full body, small face, side face, long-range perspective, glasses, sun glasses,back, bad hands, missing fingers, one hand with more than 5 fingers"

    # read json
    result_data = []
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        prompt = pos_prefix + ',' + benchmark_json[key]['prompt'] + ',' + pos_suffix

        negative_prompt = neg_prefix + ',' + benchmark_json[key]['negative_prompt']
        characters = benchmark_json[key]['characters']

        character_1_prompt = characters[0]['prompt']
        character_1_negative_prompt = characters[0]['negative_prompt']
        character_1_image_path = characters[0]['image_path']

        character_2_prompt = characters[1]['prompt']
        character_2_negative_prompt = characters[1]['negative_prompt']
        character_2_image_path = characters[1]['image_path']

        '''
        start to generate image!
        '''
        print("-" * 30, "start to gen image,", key)

        # base image
        input_prompt = [[prompt, prompt],
                        [(character_1_prompt, character_1_negative_prompt, character_1_image_path),
                         (character_2_prompt, character_2_negative_prompt, character_2_image_path)]
                        ]
        image = sample_image(
            pipe,
            input_prompt=input_prompt,  # len(input_prompt)=2 len(input_prompt[0])=2
            concept_models=pipe_concepts,
            input_neg_prompt=[negative_prompt] * len(input_prompt),
            generator=torch.Generator(device).manual_seed(args.seed),
            controller=controller,
            face_app=face_app,
            controlnet_conditioning_scale=args.IdentityNet_rate,
            stage=1,
            guidance_scale=args.cfg_scale,
            **kwargs)

        # save loop1 image
        print("-" * 30, "save loop 1 image,")
        tmp_name = os.path.join(save_dir, f'{key}_loop1.png')
        image[0].save(tmp_name)

        mask1 = predict_mask(detect_model, sam, image[0], 'male')
        mask2 = predict_mask(detect_model, sam, image[0], 'female')

        if mask1 is not None or mask2 is not None:
            face_info = face_app.get(cv2.cvtColor(np.array(image[0]), cv2.COLOR_RGB2BGR))
            face_kps = draw_kps_multi(image[0], [face['kps'] for face in face_info])

            image = sample_image(
                pipe,
                input_prompt=input_prompt,
                concept_models=pipe_concepts,
                input_neg_prompt=[negative_prompt] * len(input_prompt),
                generator=torch.Generator(device).manual_seed(args.seed),
                controller=controller,
                face_app=face_app,
                image=face_kps,
                stage=2,
                controlnet_conditioning_scale=args.IdentityNet_rate,
                region_masks=[mask1, mask2],
                guidance_scale=args.cfg_scale,
                **kwargs)

        else:
            print("no detect man and woman face")

        save_image_path = os.path.join(save_dir, f'{key}_result.png')
        image[1].save(save_image_path)
        result_data.append({'generated_image': os.path.abspath(save_image_path),
                            'character_image_path_list': [character_1_image_path, character_2_image_path],
                            'caption': prompt})

    result_json_path = os.path.join(save_dir, 'result.json')
    with open(result_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(result_data, f_out, sort_keys=True, indent=4, ensure_ascii=False)
