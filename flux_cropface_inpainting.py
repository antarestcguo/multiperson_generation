import torch
import numpy as np
import os
import argparse
from typing import Tuple, List
from PIL import Image, ImageDraw
import copy
import json
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline, FluxPipeline
import cv2
from diffusers.utils import load_image

from utils.baseline_sep_utils import build_dino_segment_model, draw_kps_multi
from utils.inpainting_utils import filter_mask, predict_face_mask, load_instant_inpainting_model, build_face_app, \
    crop_face_inpainting
from utils.match_utils import match_body_face, match_face_body, person_detector, match_character_t2i, load_clip


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    # general param
    parser.add_argument('--seed', default=52, type=int)
    # save path
    parser.add_argument('--save_dir', default='./results', type=str)

    # json path
    parser.add_argument('--benchmark_file_path', default='./benchmark/benchmark.json')
    # base model path
    parser.add_argument('--flux_model', default='./checkpoint/Flux', type=str)
    # instantID
    parser.add_argument('--pretrained_model', default='./checkpoint/RealVisXL', type=str)
    parser.add_argument('--controlnet_path', default='./checkpoint/InstantID/ControlNetModel', type=str)
    parser.add_argument('--face_adapter_path', default='./checkpoint/InstantID/ip-adapter.bin', type=str)
    parser.add_argument('--antelopev2_path', default='./checkpoint/antelopev2', type=str)

    # instant ID param
    parser.add_argument('--cfg_scale', default=7.5, type=float)
    parser.add_argument('--IdentityNet_rate', default=0.8, type=float)
    parser.add_argument('--adapter_ratio', default=0.8, type=float)
    parser.add_argument('--controlNet_ratio', default=0.8, type=float)
    parser.add_argument("--face_strength", default=0.7, type=float)
    parser.add_argument("--loop2_face_strength", default=0.0, type=float)

    # other control
    parser.add_argument('--spatial_condition', default='', type=str)
    parser.add_argument('--t2i_controlnet_path', default='', type=str)
    parser.add_argument('--style_lora', default='', type=str)

    # SEG
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)

    # debug
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # make base folder
    save_dir = args.save_dir
    benchmark_file_path = args.benchmark_file_path
    os.makedirs(save_dir, exist_ok=True)

    # base param
    base_seed = args.seed
    max_gen_num = 4
    face_prob_thr = 0.65
    face_area_thr = 64.0 * 64.0
    mask_iou_thr = 0.3
    face_strength = args.face_strength
    loop2_face_strength = args.loop2_face_strength
    face_filter_kernal = 5
    color_list = ['yellow', 'pink', 'green']
    b_crop_inpaiting = True

    torch_dtype = torch.float16
    model_type = "fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load flux model
    pipe = FluxPipeline.from_pretrained(args.flux_model,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=True,
                                        variant=model_type).to(device)
    pipe.enable_model_cpu_offload()

    # load instantid model
    instant_inpainting_pipe = load_instant_inpainting_model(pretrained_model=args.pretrained_model,
                                                            controlnet_path=args.controlnet_path,
                                                            face_adapter=args.face_adapter_path,
                                                            adapter_ratio=args.adapter_ratio, device=device)

    # load other model
    detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)
    face_app = build_face_app(args.antelopev2_path)
    clip_model, clip_processor = load_clip(device)

    # gen settings
    pos_prefix = "masterpiece, deblurring, close-up front portrait photo, Detailed clear face, frontal face, Asian"
    pos_suffix = "realistic, photographic, masterpiece, best-quality, intricate detail"
    # pos_prefix = "full-body, frontal body view, clear face, reasonable limb hand finger structure, realistic photo"
    # pos_suffix = "photographic, masterpiece, best-quality, intricate detail"

    char_pos_prefix = "look straight ahead"
    char_negative_prefix = "Roll your eyes"
    num_inference_steps = 50
    guidance_scale = args.cfg_scale

    # read json
    result_data = []
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        prompt = benchmark_json[key]['prompt']

        negative_prompt = benchmark_json[key]['negative_prompt']
        characters = benchmark_json[key]['characters']

        character_1_prompt = characters[0]['prompt']
        character_1_negative_prompt = char_negative_prefix + "," + characters[0]['negative_prompt']
        character_1_image_path = characters[0]['image_path']

        character_2_prompt = characters[1]['prompt']
        character_2_negative_prompt = char_negative_prefix + "," + characters[1]['negative_prompt']
        character_2_image_path = characters[1]['image_path']

        input_prompt = pos_prefix + ',' + prompt + ',' + pos_suffix
        # input_negative_prompt = neg_prefix + ',' + negative_prompt

        '''
                start to generate image!
        '''
        print("-" * 30, "start to gen image,", key)
        b_save = False
        base_seed = args.seed

        # read char image and embedding
        ref_face_image_1 = load_image(character_1_image_path)
        face_info_char1 = face_app.get(cv2.cvtColor(np.array(ref_face_image_1), cv2.COLOR_RGB2BGR))
        # face_info_char1 = \
        #     sorted(face_info_char1, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        #         -1]
        face_info_char1 = sorted(face_info_char1, key=lambda x: x['det_score'], reverse=True)[0]
        face_emb_char1 = face_info_char1["embedding"]

        ref_face_image_2 = load_image(character_2_image_path)
        face_info_char2 = face_app.get(cv2.cvtColor(np.array(ref_face_image_2), cv2.COLOR_RGB2BGR))
        # face_info_char2 = \
        #     sorted(face_info_char2, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        #         -1]
        face_info_char2 = sorted(face_info_char2, key=lambda x: x['det_score'], reverse=True)[0]

        face_emb_char2 = face_info_char2["embedding"]

        if args.debug:
            # save character
            tmp_name = os.path.join(save_dir, f'{key}_ref_char1_face.jpg')
            ref_face_image_1.crop(face_info_char1['bbox']).save(tmp_name)
            tmp_name = os.path.join(save_dir, f'{key}_ref_char2_face.jpg')
            ref_face_image_2.crop(face_info_char2['bbox']).save(tmp_name)

        for gen_time in range(max_gen_num):
            base_seed += 1
            images = pipe(
                prompt=input_prompt,
                num_inference_steps=num_inference_steps, num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device).manual_seed(base_seed),
            ).images

            # start to check
            gen_image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            face_info = face_app.get(gen_image)

            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) *
                                                        (x['bbox'][3] - x['bbox'][1]))[::-1]

            # draw face kps
            face_kps = draw_kps_multi(images[0], [face['kps'] for face in face_info])

            face_bboxes = []
            for it_face in face_info:
                face_w = it_face['bbox'][2] - it_face['bbox'][0]
                face_h = it_face['bbox'][3] - it_face['bbox'][1]
                area = face_w * face_h
                face_prob = it_face['det_score']

                if face_prob < face_prob_thr or area < face_area_thr:
                    continue

                print(f"init face wxh: {face_w}*{face_h}={area}", "prob", face_prob)

                face_bboxes.append(it_face['bbox'])

            if len(face_bboxes) < 2:
                print("face invalid, re-generate base image")
                continue

            face_bboxes = face_bboxes[:3]

            # det body
            body_boxes = person_detector(detect_model, images[0], box_threshold=0.5,
                                         text_threshold=0.25)
            print("init body_boxes", len(body_boxes))
            # match face and body
            body_boxes = match_face_body(face_bboxes, body_boxes)

            # save loop1 image
            print("-" * 30, "save loop 1 image,")
            #  match t2i
            char_text_list = [character_1_prompt, character_2_prompt]
            res_body_box = match_character_t2i(images[0], char_text_list, detect_model, sam, clip_model, clip_processor,
                                               device, person_boxes=body_boxes)

            face_res = match_body_face(res_body_box, face_bboxes)

            # loop2 instantid re-draw
            face_masks = predict_face_mask(sam, images[0], face_res)

            # save masks
            mask_init_char1 = Image.fromarray((face_masks[0].cpu().numpy() * 255).astype(np.uint8))
            mask_init_char2 = Image.fromarray((face_masks[1].cpu().numpy() * 255).astype(np.uint8))
            mask_char1 = filter_mask(mask_init_char1, kernal_size=face_filter_kernal, gaussian_kernal=15)
            mask_char2 = filter_mask(mask_init_char2, kernal_size=face_filter_kernal, gaussian_kernal=15)

            # filter face kps
            face_kps_char1 = np.expand_dims(np.array(mask_char1) / 255, 2).repeat(3, axis=2).astype(
                np.uint8) * np.array(
                face_kps)
            face_kps_char2 = np.expand_dims(np.array(mask_char2) / 255.0, 2).repeat(3, axis=2).astype(
                np.uint8) * np.array(face_kps)

            # inpainting
            print("crop inpainting", key)
            face_inpainting_1 = crop_face_inpainting(images[0].copy(), face_app, sam, face_res[0],
                                                     instant_inpainting_pipe,
                                                     char_pos_prefix + "," + character_1_prompt + "," + char_pos_prefix,
                                                     character_1_negative_prompt,
                                                     face_emb_char1,
                                                     face_strength, device, base_seed,
                                                     )
            face_inpainting_2 = crop_face_inpainting(face_inpainting_1.copy(), face_app, sam, face_res[1],
                                                     instant_inpainting_pipe,
                                                     char_pos_prefix + "," + character_2_prompt + "," + char_pos_prefix,
                                                     character_2_negative_prompt,
                                                     face_emb_char2,
                                                     face_strength, device, base_seed,
                                                     )
            if loop2_face_strength > 0:
                face_inpainting_1 = crop_face_inpainting(face_inpainting_2.copy(), face_app, sam, face_res[0],
                                                         instant_inpainting_pipe,
                                                         "a person, masterpiece, high quality",
                                                         "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, deformed, glitch,noisy",
                                                         face_emb_char1,
                                                         loop2_face_strength, device, base_seed)

                face_inpainting_2 = crop_face_inpainting(face_inpainting_1.copy(), face_app, sam, face_res[1],
                                                         instant_inpainting_pipe,
                                                         "a person, masterpiece, high quality",
                                                         "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, deformed, glitch,noisy",
                                                         face_emb_char2,
                                                         loop2_face_strength, device, base_seed)
            # save result
            save_image_path = os.path.join(save_dir, f'{key}_result.png')
            face_inpainting_2.save(save_image_path)

            if args.debug:
                # draw match and save
                draw = ImageDraw.Draw(images[0])
                # body
                for idx, it_bbox in enumerate(res_body_box):
                    draw.rectangle(it_bbox.tolist(), outline=color_list[idx], width=5)
                # face
                for idx, it_bbox in enumerate(face_res):
                    draw.rectangle(it_bbox.tolist(), outline=color_list[idx], width=5)
                save_img_name = os.path.join(save_dir, f'{key}_loop1.png')
                images[0].save(save_img_name)

            b_save = True
            print("gen times:", gen_time)
            break

        if b_save is False:
            # save error
            print("no face to swap")
            save_image_path = os.path.join(save_dir, f'{key}_result.png')
            images[0].save(save_image_path)

        result_data.append({'generated_image': os.path.abspath(save_image_path),
                            'character_image_path_list': [character_1_image_path, character_2_image_path],
                            'caption': benchmark_json[key]['prompt'],
                            })
    result_json_path = os.path.join(save_dir, 'result.json')
    with open(result_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(result_data, f_out, sort_keys=True, indent=4, ensure_ascii=False)
