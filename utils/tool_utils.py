import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import onnxruntime
import torch
import torchvision
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from diffusers import AutoPipelineForInpainting


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1
    return out_image, out_mask


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def segment_anything_groundingDINO(source_path,
                                   mask_path,
                                   rgba_path,
                                   rgb_path,
                                   mode):
    SAM_ENCODER_VERSION = "vit_h"

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "models/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    CLASSES = ["human"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    for image_name in tqdm(os.listdir(source_path)):
        image = cv2.imread(source_path+image_name)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        predict_mask = detections.mask

        # sometimes need to flip mask
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        mask[predict_mask[0] == True, 0] = 255
        # mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
        # mask[predict_mask[0] == True, 0] = 0

        if mode == 'dataset':
            rgba_image = np.concatenate((image, mask), 2)
            mask_image = cv2.dilate(np.repeat(mask, 3, 2), np.ones((5, 5)), iterations=8)
            cv2.imwrite(rgba_path + "{}.png".format(image_name.split('.')[0]), rgba_image)
            cv2.imwrite(mask_path + "{}.png".format(image_name.split('.')[0]), mask_image)
        elif mode == 'rgba':
            rgba_image = np.concatenate((image, mask), 2)
            cv2.imwrite(rgba_path + "{}.png".format(image_name.split('.')[0]), rgba_image)
        else:
            mask_image = np.repeat(mask, 3, 2)
            mask = (mask == 255).astype(np.uint8)
            rgb_image = mask * image
            cv2.imwrite(rgb_path + "{}.png".format(image_name.split('.')[0]), rgb_image)
            cv2.imwrite(mask_path + "{}.png".format(image_name.split('.')[0]), mask_image)


def foreground_background_paste(background_path,
                                rgb_path,
                                mask_path,
                                result_path):
    for img in tqdm(os.listdir(rgb_path)):
        from_name = img.split('_to_')[0] + '.png'
        foreground = Image.open(rgb_path + img).convert('RGB')
        background = Image.open(background_path + from_name)
        mask = np.array(Image.open(mask_path + img))

        mask = (mask == 255).astype(np.uint8)
        background = np.array(background) * (1 - mask)
        result = Image.fromarray(np.array(foreground) + background)
        result.save(result_path + img)


def inpainting_lama(source_path,
                    mask_path,
                    result_path):
    model = onnxruntime.InferenceSession('models/lama_fp32_1024.onnx',
                                         providers=['TensorrtExecutionProvider',
                                                    'CUDAExecutionProvider',
                                                    'CPUExecutionProvider'])
    h, w = 1024, 1024
    for img in tqdm(os.listdir(mask_path)):
        input_mask = Image.open(mask_path + img).convert('L').resize((h, w))
        input_source = Image.open(source_path + img).resize((h, w))
        image, mask = prepare_img_and_mask(input_source, input_mask, 'cpu')
        outputs = model.run(None, {'image': image.numpy().astype(np.float32), 'mask': mask.numpy().astype(np.float32)})
        output = outputs[0][0].transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(output).save(result_path + img)


def inpainting_diffusion(source_path,
                         mask_path,
                         result_path):
    for img in tqdm(os.listdir(mask_path)):
        pipe = AutoPipelineForInpainting.from_pretrained("models/stable-diffusion-xl-1.0-inpainting-0.1",
                                                         torch_dtype=torch.float16, variant="fp16").to("cuda")
        input_mask = Image.open(mask_path + img)
        input_image = Image.open(source_path + img)
        generator = torch.Generator(device="cuda").manual_seed(0)
        image = pipe(
            prompt="",
            negative_prompt="",
            image=input_image,
            mask_image=input_mask,
            guidance_scale=8.0,
            num_inference_steps=20,
            strength=0.6,
            generator=generator,
        ).images[0]
        image.save(result_path + img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--mask_path', type=str, default='')
    parser.add_argument('--rgba_path', type=str, default='')
    parser.add_argument('--rgb_path', type=str, default='')
    parser.add_argument('--background_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='', choices=['dataset', 'rgba', 'rgb', 'paste', 'inpainting_lama', 'inpainting_diffusion'])
    args = parser.parse_args()

    if args.mode == 'paste':
       foreground_background_paste(args.background_path,
                                   args.rgb_path,
                                   args.mask_path,
                                   args.result_path)

    elif args.mode == 'inpainting_lama':
        inpainting_lama(args.source_path,
                        args.mask_path,
                        args.background_path)

    elif args.mode == 'inpainting_diffusion':
        inpainting_diffusion(args.source_path,
                             args.mask_path,
                             args.background_path)

    else:
        segment_anything_groundingDINO(args.source_path,
                                       args.mask_path,
                                       args.rgba_path,
                                       args.rgb_path,
                                       args.mode)