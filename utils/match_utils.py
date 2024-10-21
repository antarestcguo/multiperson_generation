import clip
from segment_anything import build_sam, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict
import torch
import groundingdino.datasets.transforms as T
import numpy as np
from typing import Tuple, List
from torchvision.ops import box_convert
from PIL import Image, ImageDraw
from groundingdino.util import box_ops


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


def compute_intersection_area(bbox1, bbox2):  # xyxy, real coor
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])

    if xmax - xmin < 0 or ymax - ymin < 0:
        return 0
    else:
        return (xmax - xmin) * (ymax - ymin)


def compute_iou(bbox1, bbox2):
    I = compute_intersection_area(bbox1, bbox2)
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    IOU = I / (w1 * h1 + w2 * h2 - I)
    return IOU


def match_body_face(body_bbox, face_bbox):
    face_res = []

    for it_body in body_bbox:
        max_iou = -1
        select_idx = -1
        for idx, it_face in enumerate(face_bbox):
            iou = compute_iou(it_body, it_face)
            if iou > max_iou:
                max_iou = iou
                select_idx = idx
        face_res.append(face_bbox[select_idx])
        del face_bbox[select_idx]

    return face_res


def match_face_body(face_bboxes, body_bboxes):
    res_body_bboxes = []
    for it_face in face_bboxes:
        max_iou = 0
        idx = 0
        for i, it_body in enumerate(body_bboxes):
            iou = compute_iou(it_face, it_body)
            if iou >= max_iou:
                max_iou = iou
                idx = i
        res_body_bboxes.append(body_bboxes[idx])
        del body_bboxes[idx]

    return res_body_bboxes


def person_detector(model,
                    PILimage,
                    box_threshold,
                    text_threshold):
    # detected_obj = []
    image_source, image = load_image_dino(PILimage)
    h, w, _ = image_source.shape
    # det human
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption="person",
        box_threshold=0.3,
        text_threshold=0.25
    )
    # boxes = boxes * torch.tensor([w, h, w, h])
    H_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
    # boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # if len(boxes) > 0:
    #     boxes = [list(map(int, box)) for box in boxes]
    #     detected_obj.extend(boxes)

    detected_obj = [it.cpu().numpy() for it in H_boxes_xyxy]
    return detected_obj


def match_character_t2i(generate_image, char_text_list, dino_model, sam_predictor, clip_model, clip_processor, device,
                        person_boxes=None):
    box_threshold = 0.5,
    text_threshold = 0.25
    res_bbox = []

    if not person_boxes:
        person_boxes = person_detector(
            dino_model, generate_image, box_threshold, text_threshold)
        print("det person num:", len(person_boxes))

    person_image_list = []
    for person_box in person_boxes:
        person_image = generate_image.crop(tuple(person_box))
        person_image_list.append(person_image)

    template_text = ['male, man, gentelman, boy', 'female, woman, lady, girl']
    cos = torch.nn.CosineSimilarity(dim=1)
    for character_text in char_text_list:
        tmp_character_input = clip.tokenize(character_text).to(device)
        template_text_input_male = clip.tokenize(template_text[0]).to(device)
        template_text_input_female = clip.tokenize(template_text[1]).to(device)

        template_prob_male = cos(clip_model.encode_text(tmp_character_input),
                                 clip_model.encode_text(template_text_input_male))
        template_prob_female = cos(clip_model.encode_text(tmp_character_input),
                                   clip_model.encode_text(template_text_input_female))

        character_input = template_text_input_male if template_prob_male > template_prob_female else template_text_input_female

        character_similarity = 0
        select_idx = -1
        for idx, person_image in enumerate(person_image_list):
            person_input = clip_processor(person_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(person_input, character_input)
                probs = logits_per_image.cpu().numpy().tolist()
                similarity = probs[0][0]
            if similarity > character_similarity:
                character_similarity = similarity
                select_idx = idx
        res_bbox.append(person_boxes[select_idx])
        del person_boxes[select_idx]
        del person_image_list[select_idx]
    return res_bbox


def load_clip(device):
    # clip_model, clip_processor = clip.load("ViT-B/32", device)
    clip_model, clip_processor = clip.load("./checkpoint/clip/ViT-B-32.pt", device)
    return clip_model, clip_processor
