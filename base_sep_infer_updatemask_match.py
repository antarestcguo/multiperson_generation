import torch
import numpy as np
import os
import argparse
from typing import Tuple, List
from PIL import Image, ImageDraw
import copy
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from insightface.app import FaceAnalysis
import json

import cv2
import math
from diffusers.utils import load_image

from utils.baseline_sep_utils import sample_image, build_model_sd, build_dino_segment_model, draw_kps_multi
from utils.inpainting_utils import predict_face_mask
from utils.match_utils import match_body_face, match_face_body, person_detector, match_character_t2i, load_clip


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    # general param
    parser.add_argument('--seed', default=53, type=int)
    # save path
    parser.add_argument('--save_dir', default='./results', type=str)

    # json path
    parser.add_argument('--benchmark_file_path', default='./benchmark/benchmark.json')

    # instantID
    parser.add_argument('--pretrained_model', default='./checkpoint/wildcardxXL', type=str)
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
    base_seed = args.seed
    max_gen_num = 4
    face_prob_thr = 0.65
    face_area_thr = 64.0 * 64.0
    mask_iou_thr = 0.3

    prompts_tmp = ["aaa"] * 2
    # load model
    pipe, controller, pipe_concepts, face_app = build_model_sd(args.pretrained_model, args.controlnet_path,
                                                               args.face_adapter_path, device, prompts_tmp,
                                                               args.antelopev2_path, width // 32, height // 32,
                                                               args.style_lora, args.t2i_controlnet_path,
                                                               args.adapter_ratio)

    detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)
    clip_model, clip_processor = load_clip(device)

    # setting
    pos_prefix = "masterpiece, close-up front portrait photo, half-length photo, Asian"
    tmp_pos_prefix = "masterpiece, front portrait photo, half-length photo, Asian"
    pos_suffix = "photographic, reasonable body structure"
    # neg_prefix = "back view, small face, side face, long-range perspective, glasses, sun glasses, bad hands, missing fingers, one hand with more than 5 fingers"
    neg_prefix = "back view person, extra limb, missing limb, floating limbs, mutated hands, extra fingers, missing fingers, disconnected limbs"
    neg_suffix = "disfigured, unreal, deformed, distorted, low quality, ugly, disgusting, blurry, low quality, bad"

    character_pos_prefix = "masterpiece, close-up front portrait photo"
    tmp_character_pos_prefix = "masterpiece, front portrait photo"

    # read json
    result_data = []
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        if benchmark_json[key]['prompt'].find("Close-up") != -1 or benchmark_json[key]['prompt'].find("close-up") != -1:
            prompt = tmp_pos_prefix + ',' + benchmark_json[key]['prompt'] + ',' + pos_suffix
        else:
            prompt = pos_prefix + ',' + benchmark_json[key]['prompt'] + ',' + pos_suffix

        negative_prompt = neg_prefix + ',' + benchmark_json[key]['negative_prompt'] + ',' + neg_suffix
        characters = benchmark_json[key]['characters']

        if characters[0]['prompt'].find("Close-up") != -1 or characters[0]['prompt'].find("close-up") != -1:
            character_1_prompt = tmp_character_pos_prefix + ',' + characters[0]['prompt']
        else:
            character_1_prompt = characters[0]['prompt']
        character_1_negative_prompt = neg_prefix + ',' + characters[0]['negative_prompt']
        character_1_image_path = characters[0]['image_path']

        if characters[1]['prompt'].find("Close-up") != -1 or characters[1]['prompt'].find("close-up") != -1:
            character_2_prompt = tmp_character_pos_prefix + ',' + characters[1]['prompt']
        else:
            character_2_prompt = characters[1]['prompt']
        character_2_negative_prompt = neg_prefix + ',' + characters[1]['negative_prompt']
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
        b_save = False
        base_seed = args.seed
        for gen_time in range(max_gen_num):
            base_seed += 1
            image = sample_image(
                pipe,
                input_prompt=input_prompt,  # len(input_prompt)=2 len(input_prompt[0])=2
                concept_models=pipe_concepts,
                input_neg_prompt=[negative_prompt] * len(input_prompt),
                generator=torch.Generator(device).manual_seed(base_seed),
                controller=controller,
                face_app=face_app,
                controlnet_conditioning_scale=args.IdentityNet_rate,
                stage=1,
                guidance_scale=args.cfg_scale,
                **kwargs)

            # check face and body
            gen_image = cv2.cvtColor(np.array(image[0]), cv2.COLOR_RGB2BGR)
            face_info = face_app.get(gen_image)

            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) *
                                                        (x['bbox'][3] - x['bbox'][1]))[::-1]

            face_bboxes = []
            for it_face in face_info:
                area = (it_face['bbox'][2] - it_face['bbox'][0]) * (it_face['bbox'][3] - it_face['bbox'][1])
                face_prob = it_face['det_score']

                if face_prob < face_prob_thr or area < face_area_thr:
                    continue

                print("init face area", area, "prob", face_prob)

                face_bboxes.append(it_face['bbox'])

            if len(face_bboxes) < 2:
                print("face invalid, re-generate base image")
                continue
            # face control
            # 后面可以过滤一下
            face_kps = draw_kps_multi(image[0], [face['kps'] for face in face_info])

            face_bboxes = face_bboxes[:3]

            # det body
            body_boxes = person_detector(detect_model, image[0], box_threshold=0.5,
                                         text_threshold=0.25)
            print("init body_boxes", len(body_boxes))
            # match face and body
            body_boxes = match_face_body(face_bboxes, body_boxes)

            #  match t2i
            char_text_list = [character_1_prompt, character_2_prompt]
            res_body_box = match_character_t2i(image[0], char_text_list, detect_model, sam, clip_model, clip_processor,
                                               device, person_boxes=body_boxes)
            face_res = match_body_face(res_body_box, face_bboxes)

            # loop2 instantid re-draw
            body_masks = predict_face_mask(sam, image[0], res_body_box)
            mask_0_cnt = body_masks[0].sum().to(torch.float)
            mask_1_cnt = body_masks[1].sum().to(torch.float)
            interaction_cnt = (body_masks[0] * body_masks[1]).to(torch.float).sum()
            iou = interaction_cnt / (mask_0_cnt + mask_1_cnt - interaction_cnt)

            print("test!!!!!!!  mask cnt:", interaction_cnt, "mask iou:", iou)
            if iou > mask_iou_thr:
                print("mask invalid, re-generate base image")
                continue

            mask_init_char1 = Image.fromarray((body_masks[0].cpu().numpy() * 255).astype(np.uint8))
            mask_init_char2 = Image.fromarray((body_masks[1].cpu().numpy() * 255).astype(np.uint8))

            image = sample_image(
                pipe,
                input_prompt=input_prompt,
                concept_models=pipe_concepts,
                input_neg_prompt=[negative_prompt] * len(input_prompt),
                generator=torch.Generator(device).manual_seed(base_seed),
                controller=controller,
                face_app=face_app,
                image=face_kps,
                stage=2,
                controlnet_conditioning_scale=args.IdentityNet_rate,
                region_masks=[body_masks[0], body_masks[1]],
                guidance_scale=args.cfg_scale,
                **kwargs)

            # save inpainting human
            save_image_path = os.path.join(save_dir, f'{key}_result.png')
            image[1].save(save_image_path)

            b_save = True
            print("gen times:", gen_time)
            break

        if b_save is False:
            # save error
            print("no face to swap")
            save_image_path = os.path.join(save_dir, f'{key}_result.png')
            image[0].save(save_image_path)

        result_data.append({'generated_image': os.path.abspath(save_image_path),
                            'character_image_path_list': [character_1_image_path, character_2_image_path],
                            'caption': benchmark_json[key]['prompt']})

        result_json_path = os.path.join(save_dir, 'result.json')
        with open(result_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(result_data, f_out, sort_keys=True, indent=4, ensure_ascii=False)
