import cv2
import numpy as np
import PIL.Image as Image
import os
import torch
from groundingdino.util.slconfig import SLConfig
from segment_anything import build_sam, SamPredictor
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from typing import Tuple, List
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import annotate, predict
from groundingdino.util import box_ops
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLPipeline, ControlNetModel

from src.inpainting_pipelines.instantid_inpainting import InstantidSingleConceptPipeline
from .baseline_sep_utils import draw_kps_multi

def filter_mask(mask_image, kernal_size=5, gaussian_kernal=0):  # mask is PIL.Image
    # gen kernal
    kernel = np.ones((int(kernal_size), int(kernal_size)), np.uint8)
    # convert Image to numpy
    mask = np.array(mask_image)

    # fill hole
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 使用cv.RETR_CCOMP寻找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
    # 绘制轮廓内部
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, 255, -1)

    # dilated
    filtered_mask = cv2.dilate(mask, kernel, iterations=1)

    # gaussian
    if gaussian_kernal > 0:
        filtered_mask = cv2.GaussianBlur(filtered_mask, (gaussian_kernal, gaussian_kernal), 0)

    # convert back to Image
    image_mask = Image.fromarray(filtered_mask)
    return image_mask


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
    masks, prob, _ = sam.predict_torch( #masks, iou_predictions, low_res_masks
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.squeeze(1)
    prob = prob.squeeze(1)
    return masks, prob

def predict_face_mask(sam, PILimage, face_bboxes):
    image_source, image = load_image_dino(PILimage)
    sam.set_image(image_source)

    res_face_box = torch.cat([torch.from_numpy(it) for it in face_bboxes]).reshape(-1, 4)
    transformed_boxes = sam.transform.apply_boxes_torch(res_face_box,
                                                        image_source.shape[:2]).cuda()  # boxes_xyxy to cuda,

    masks, iou_predictions, low_res_masks = sam.predict_torch(  # masks, iou_predictions, low_res_masks
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.squeeze(1)


def build_face_app(model_path):
    app = FaceAnalysis(name='antelopev2',
                       root=model_path,
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    return app

def load_instant_inpainting_model(pretrained_model, controlnet_path, face_adapter, adapter_ratio, device):
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

    return pipe_concept

def crop_img(bbox, ori_img):  # PIL.Image, return PILImage:crop_img,crop_coor,align_coor
    w, h = ori_img.size
    ori_x1 = int(bbox[0])
    ori_y1 = int(bbox[1])
    ori_x2 = int(bbox[2])
    ori_y2 = int(bbox[3])

    ori_w = ori_x2 - ori_x1
    ori_h = ori_y2 - ori_y1

    crop_x1 = int(max(0, ori_x1 - ori_w))
    crop_y1 = int(max(0, ori_y1 - ori_h))
    crop_x2 = int(min(w, ori_x2 + ori_w))
    crop_y2 = int(min(h, ori_y2 + ori_h))

    crop_img = ori_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    align_x1 = ori_x1 - crop_x1
    align_y1 = ori_y1 - crop_y1
    align_x2 = ori_x2 - crop_x1
    align_y2 = ori_y2 - crop_y1

    return crop_img, (crop_x1, crop_y1, crop_x2, crop_y2), (align_x1, align_y1, align_x2, align_y2)

def resize_img(
        input_image,
        max_side=1280,
        min_side=1024,
        size=None,
        pad_to_max_side=False,
        mode=Image.BILINEAR,
        base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
        ratio = -1
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
        offset_y: offset_y + h_resize_new, offset_x: offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image, ratio


def crop_face_inpainting(ori_img, face_app, sam, bbox, inpainting_pipeline, prompt, negative_prompt, embedding,
                         strength, device, seed, key=None, save_dir=None, char_idx=None):
    # crop image
    crop_image, crop_coor, align_coor = crop_img(bbox, ori_img)
    ori_w, ori_h = crop_image.size

    # resize
    resize_image, ratio = resize_img(crop_image)

    resize_w, resize_h = resize_image.size

    # det
    gen_image = cv2.cvtColor(np.array(resize_image), cv2.COLOR_RGB2BGR)
    face_info = face_app.get(gen_image)

    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) *
                                                (x['bbox'][3] - x['bbox'][1]))[::-1]
    idx = -1
    for i, it in enumerate(face_info):
        new_bbox = it['bbox']
        if new_bbox[0] < resize_w / 2 < new_bbox[2] and new_bbox[1] < resize_h / 2 < new_bbox[3]:
            idx = i
            break
    if idx == -1:
        print("error")

    # draw new face kps
    face_kps = draw_kps_multi(resize_image, [face_info[idx]['kps']])

    # sam seg
    mask = predict_face_mask(sam, resize_image, [face_info[idx]['bbox']])[0]
    mask_init = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
    # gaussian_kernel = compute_gaussian_kernel(bbox=face_info[idx]['bbox'])
    mask_char = filter_mask(mask_init, kernal_size=5, gaussian_kernal=15)

    # inpainting
    face_inpainting = inpainting_pipeline(prompt=prompt,
                                          negative_prompt=negative_prompt,
                                          image=face_kps,
                                          image_embeds=embedding,
                                          mask_image=mask_char,
                                          ori_image=resize_image,
                                          strength=strength,
                                          generator=torch.Generator(device).manual_seed(seed),
                                          # guidance_scale=3.0,
                                          ).images[0]

    if key is not None and save_dir is not None and char_idx is not None:
        save_name = os.path.join(save_dir, f'{key}_result_crop_mask_{char_idx}.png')
        mask_char.save(save_name)
        save_name = os.path.join(save_dir, f'{key}_result_resize_inpainting_{char_idx}.png')
        face_inpainting.save(save_name)

    # re-align to original image
    image_back, _ = resize_img(face_inpainting, size=(ori_w, ori_h))

    # paste
    ori_img.paste(image_back, (crop_coor[0], crop_coor[1]))

    return ori_img
