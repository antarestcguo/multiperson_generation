import argparse
import json
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict
import cv2
from torchvision.ops import box_convert
import torchvision
from PIL import Image
import numpy as np
import clip
import insightface
from segment_anything import build_sam, SamPredictor


def person_detector(model,
                    image_path,
                    box_threshold,
                    text_threshold):
    image_source, image = load_image(image_path)

    detected_obj = []
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption='person',
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    if len(boxes) > 0:
        boxes = [list(map(int, box)) for box in boxes]
        detected_obj.extend(boxes)
    return detected_obj


def face2face_similarity(insightface_model, generate_image_path, character_image_path_list):
    output_face_similarity = 0
    generate_image = cv2.imdecode(np.fromfile(
        generate_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    generate_image_faces = insightface_model.get(generate_image)
    character_faces = []
    for character_image_path in character_image_path_list:
        character_image = cv2.imdecode(np.fromfile(
            character_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        faces = insightface_model.get(character_image)
        if len(faces) > 0:
            face = sorted(faces, key=lambda x: x['det_score'], reverse=True)[0]
            character_faces.append(face)

    for character_face in character_faces:
        character_face_embedding = np.array(
            character_face.normed_embedding).reshape((1, -1))
        character_similarity = 0
        for generate_image_face in generate_image_faces:
            generate_face_embedding = np.array(
                generate_image_face.normed_embedding).reshape((1, -1))
            similarity = np.dot(generate_face_embedding,
                                character_face_embedding.T)[0][0]
            if similarity > character_similarity:
                character_similarity = similarity
        output_face_similarity += character_similarity
    if len(character_faces) > 0:
        output_face_similarity = output_face_similarity / len(character_faces)
    else:
        output_face_similarity = 0
    return output_face_similarity


def segment_and_crop_person_image(sam_predictor, image, boxes_in, point_coords=None, point_labels=None, device='cuda'):
    image = np.asarray(image)
    sam_predictor.set_image(image)
    boxes = torch.Tensor(boxes_in).to(device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes, image.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    mask = masks[0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    person_image = cv2.bitwise_and(image, image, mask=mask[0])
    person_image = Image.fromarray(person_image)
    person_image.crop(tuple(boxes_in[0]))
    return person_image


def char2char_similarity(dino_model,
                         clip_model,
                         sam_predictor,
                         processor,
                         generate_image_path,
                         character_image_path_list,
                         device,
                         box_threshold=0.5,
                         text_threshold=0.25):
    output_similarity = 0
    generate_image = Image.open(generate_image_path).convert('RGB')
    person_boxes = person_detector(
        dino_model, generate_image_path, box_threshold, text_threshold)
    person_image_list = []
    for person_box in person_boxes:
        person_image = generate_image.crop(tuple(person_box))
        person_image_list.append(person_image)

    character_similarity_list = []
    for character_image_path in character_image_path_list:
        character_image = Image.open(character_image_path)
        person_box = person_detector(
            dino_model, character_image_path, box_threshold, text_threshold)[0]
        character_image = character_image.crop(tuple(person_box))
        character_input = processor(character_image).unsqueeze(0).to(device)
        character_similarity = 0
        with torch.no_grad():
            character_features = clip_model.encode_image(character_input)
        for person_image in person_image_list:
            person_input = processor(person_image).unsqueeze(0).to(device)
            with torch.no_grad():
                person_features = clip_model.encode_image(person_input)
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(character_features[0], person_features[0]).item()
            if similarity > character_similarity:
                character_similarity = similarity
        character_similarity_list.append(character_similarity)
        output_similarity += character_similarity
    output_similarity = output_similarity / len(character_image_path_list)
    return output_similarity


def text2img_similarity(model,
                        processor,
                        image_path,
                        caption,
                        device):
    generation_image = Image.open(image_path).convert('RGB')

    image = processor(generation_image).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.cpu().numpy().tolist()
    return probs[0][0]


class Evaluator:
    def __init__(self):
        self.insightface_model = insightface.app.FaceAnalysis(
            # root='./',
            allowed_modules=None,
            providers=['CUDAExecutionProvider'])
        # self.insightface_model = insightface.app.FaceAnalysis(name='antelopev2',
        #                                                       root="./checkpoint/antelopev2",
        #                                                       providers=['CUDAExecutionProvider',
        #                                                                  'CPUExecutionProvider'])
        self.insightface_model.prepare(
            ctx_id=0, det_thresh=0.5, det_size=(512, 512))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dino_model = load_model(
            "./checkpoint/GroundingDINO/GroundingDINO_SwinT_OGC.py",
            "./checkpoint/GroundingDINO/groundingdino_swint_ogc.pth")
        self.clip_model, self.clip_processor = clip.load(
            "ViT-B/32", self.device)
        self.inception_model = torchvision.models.inception_v3(
            pretrained=True).to(self.device)
        sam = build_sam(
            checkpoint='./checkpoint/sam/sam_vit_h_4b8939.pth')
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def eval(self, generate_image_path, character_image_path_list, caption):
        face_similarity = face2face_similarity(self.insightface_model,
                                               generate_image_path=generate_image_path,
                                               character_image_path_list=character_image_path_list,
                                               )

        character_similarity = char2char_similarity(self.dino_model,
                                                    self.clip_model,
                                                    self.sam_predictor,
                                                    processor=self.clip_processor,
                                                    generate_image_path=generate_image_path,
                                                    character_image_path_list=character_image_path_list,
                                                    device=self.device)

        t2i_similarity = text2img_similarity(self.clip_model, self.clip_processor,
                                             image_path=generate_image_path,
                                             caption=caption,
                                             device=self.device)
        result = {"face_similarity": face_similarity,
                  "character_similarity": character_similarity,
                  "t2i_similarity": t2i_similarity}
        return result


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument(
        '--json_path', default='/workspace/OMG/results/result.json', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    evaluator = Evaluator()
    args = parse_args()
    with open(args.json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    face_similarity = 0
    character_similarity = 0
    t2i_similarity = 0

    save_result_name = args.json_path
    save_result_name = save_result_name[:-4] + 'txt'

    with open(save_result_name, 'w') as f_result:
        for data in json_data:
            generate_image_path = data['generated_image']
            character_image_path_list = data['character_image_path_list']
            caption = data['caption']

            print("-" * 30, "eval:", generate_image_path)
            f_result.write(generate_image_path + '\n')

            result = evaluator.eval(generate_image_path,
                                    character_image_path_list, caption)
            face_similarity += result["face_similarity"]
            character_similarity += result["character_similarity"]
            t2i_similarity += result["t2i_similarity"]
            print("result: ", result)

            f_result.write("face_similarity:" + str(result["face_similarity"]) + "\t" + "character_similarity:" + str(
                result["character_similarity"]) + "\t" + "t2i_similarity:" + str(result["t2i_similarity"]) + '\n')

        print("face_similarity:", face_similarity / len(json_data))
        print("character_similarity:", character_similarity / len(json_data))
        print("text2img_similarity:", t2i_similarity / len(json_data) / 100)

        f_result.write("----------------\n")
        f_result.write("face_similarity:" + str(face_similarity / len(json_data)) + '\n')
        f_result.write("character_similarity:" + str(character_similarity / len(json_data)) + '\n')
        f_result.write("text2img_similarity:" + str(t2i_similarity / len(json_data) / 100) + '\n')
