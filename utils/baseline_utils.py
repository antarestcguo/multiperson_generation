import os
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLPipeline
# dino and sam
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
import torch
import math
import cv2
import PIL

from typing import Tuple, List

from insightface.app import FaceAnalysis

from src.pipelines.instantid_pipeline import InstantidMultiConceptPipeline
from src.pipelines.instantid_single_pieline import InstantidSingleConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from src.pipelines.instantid_pipeline import revise_regionally_controlnet_forward


def sample_image(pipe,
                 input_prompt,
                 input_neg_prompt=None,
                 generator=None,
                 concept_models=None,
                 num_inference_steps=50,
                 guidance_scale=3,
                 controller=None,
                 face_app=None,
                 image=None,
                 stage=None,
                 region_masks=None,
                 controlnet_conditioning_scale=None,
                 **extra_kargs
                 ):
    if image is not None:
        image_condition = [image]
    else:
        image_condition = None

    images = pipe(
        prompt=input_prompt,
        concept_models=concept_models,
        negative_prompt=input_neg_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": 0.8},
        controller=controller,
        image=image_condition,
        face_app=face_app,
        stage=stage,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        region_masks=region_masks,
        **extra_kargs).images
    return images


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    args = SLConfig.fromfile(ckpt_config_filename)
    model = build_model(args)
    args.device = device

    checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(filename, log))
    _ = model.eval()
    return model


def build_dino_segment_model(ckpt_repo_id, sam_checkpoint):
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return groundingdino_model, sam_predictor


def load_image_dino(image_source) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict_mask(segmentmodel, sam, image, TEXT_PROMPT):
    image_source, image = load_image_dino(image)
    boxes, logits, phrases = predict(
        model=segmentmodel,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=0.3,
        text_threshold=0.25
    )
    sam.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks[0].squeeze(0)

    return masks


def build_model_sd(pretrained_model, controlnet_path, face_adapter, device, prompts, antelopev2_path, width, height,
                   style_lora, condition_checkpoint, adapter_ratio):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = InstantidMultiConceptPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to(device)

    # controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.},
    #                               self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device, width=width, height=height,
    #                               dtype=torch.float16)
    # revise_regionally_controlnet_forward(pipe.unet, controller)
    controller = None

    controlnet_concept = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe_concept = InstantidSingleConceptPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet_concept,
        torch_dtype=torch.float16, variant="fp16"
    )
    pipe_concept.load_ip_adapter_instantid(face_adapter)
    pipe_concept.set_ip_adapter_scale(adapter_ratio)
    pipe_concept.to(device)
    pipe_concept.image_proj_model.to(pipe_concept._execution_device)

    if condition_checkpoint is not None and os.path.exists(condition_checkpoint):
        t2i_controlnet = ControlNetModel.from_pretrained(condition_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.controlnet2 = t2i_controlnet

    if style_lora is not None and os.path.exists(style_lora):
        pipe.load_lora_weights(style_lora, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')
        pipe_concept.load_lora_weights(style_lora, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')

    # modify
    app = FaceAnalysis(name='antelopev2',
                       # root=antelopev2_path,
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    return pipe, controller, pipe_concept, app


def draw_kps_multi(image_pil, kps_list,
                   color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for kps in kps_list:
        kps = np.array(kps)
        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil
