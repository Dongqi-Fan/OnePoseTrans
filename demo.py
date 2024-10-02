import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import memory_management
import cv2
import gc
import time
import torchvision
import torch
import onnxruntime
import numpy as np
import gradio as gr
import safetensors.torch as sf
import google.generativeai as genai

from typing import List
from PIL import Image
from ip_adapter import IPAdapter
from lora_wpose import train_lora
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from utils.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from pipeline_stable_diffusion_xl_oneposetrans import StableDiffusionXLOnePoseTransPipeline

# Gemini API Key
os.environ["API_KEY"] = "YOUR KEY"

# Base SDXL
base_model = 'models/RealVisXL_V4.0'

# Segment-Anything
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"

# GroundingDINO
GROUNDING_DINO_CHECKPOINT_PATH = "models/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# LaMa
lama_path = 'models/lama_fp32_1024.onnx'

# IP-Adapter
image_encoder_path = "models/IP-Adapter/image_encoder"
ip_ckpt = "models/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"

# InstantID
face_adapter = 'models/InstantID/ip-adapter.bin'

# LayerDiffuse
path_ld_diffusers_sdxl_attn = 'models/LayerDiffuse/ld_diffusers_sdxl_attn.safetensors'
path_ld_diffusers_sdxl_vae_transparent_decoder = 'models/LayerDiffuse/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
path_ld_diffusers_sdxl_vae_transparent_encoder = 'models/LayerDiffuse/ld_diffusers_sdxl_vae_transparent_encoder.safetensors'

# ControlNet
controlnet_model = 'models/controlnet-openpose-sdxl-1.0'

current_time = time.time()
formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(current_time))

models = {
	'vit_b': './models/sam_vit_b_01ec64.pth',
	'vit_l': './models/sam_vit_l_0b3195.pth',
	'vit_h': './models/sam_vit_h_4b8939.pth'
}

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]
width_gradio = 500
length_gradio = 480


def run_segmentation_dot(input_x, selected_points=[]):
    device = 'cuda'
    sam = sam_model_registry['vit_h'](checkpoint=models['vit_h']).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(input_x)

    if len(selected_points) != 0:
        points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
        labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
        transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
        print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
    else:
        transformed_points, labels = None, None

    masks, scores, logits = predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=labels,
        boxes=None,
        multimask_output=False,
    )
    masks = masks.cpu().detach().numpy()
    mask_background = np.zeros((input_x.shape[0], input_x.shape[1], 1), dtype=np.uint8)
    mask_foreground = np.ones((input_x.shape[0], input_x.shape[1], 1), dtype=np.uint8) * 255
    for ann in masks:
        mask_background[ann[0] == True, 0] = 255
        mask_foreground[ann[0] == True, 0] = 0
    gc.collect()

    foreground_image = np.concatenate((cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB), mask_background), 2)
    mask_image = cv2.dilate(np.repeat(mask_background, 3, 2), np.ones((5, 5)), iterations=8)
    path = os.path.join('results/', formatted_time)
    if not os.path.exists(path): os.mkdir(path)
    cv2.imwrite(os.path.join(path, 'mask.png'), mask_image)
    cv2.imwrite(os.path.join(path, 'foreground.png'), foreground_image)
    return np.concatenate((input_x, mask_background), 2), np.concatenate((input_x, mask_foreground), 2)


def run_segmentation_dino(input_x, mode='dataset'):
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

    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)

    CLASSES = ["human"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    detections = grounding_dino_model.predict_with_classes(
        image=input_x,
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

    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    predict_mask = detections.mask

    mask_background = np.zeros((input_x.shape[0], input_x.shape[1], 1), dtype=np.uint8)
    mask_foreground = np.ones((input_x.shape[0], input_x.shape[1], 1), dtype=np.uint8) * 255
    mask_background[predict_mask[0] == True, 0] = 255
    mask_foreground[predict_mask[0] == True, 0] = 0

    foreground_image = np.concatenate((cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB), mask_background), 2)
    path = os.path.join('results/', formatted_time)

    if mode == 'dataset':
        if not os.path.exists(path): os.mkdir(path)
        mask_image = cv2.dilate(np.repeat(mask_background, 3, 2), np.ones((5, 5)), iterations=8)
        cv2.imwrite(os.path.join(path, 'mask.png'), mask_image)
        cv2.imwrite(os.path.join(path, 'foreground.png'), foreground_image)
        return np.concatenate((input_x, mask_background), 2), np.concatenate((input_x, mask_foreground), 2)
    elif mode == 'rgba':
        cv2.imwrite(os.path.join(path, 'rgba.png'), foreground_image)
        mask_image = np.repeat(mask_background, 3, 2)
        rgb_image = (mask_image == 255).astype(np.uint8) * input_x
        return mask_image, rgb_image
    else:
        mask_image = np.repeat(mask_background, 3, 2)
        rgb_image = (mask_image == 255).astype(np.uint8) * input_x
        return mask_image, rgb_image


def run_inpainting(source_image):
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

    model = onnxruntime.InferenceSession(lama_path, providers=['TensorrtExecutionProvider',
                                                               'CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
    h, w = 1024, 1024
    source_image = Image.fromarray(source_image).resize((h, w))
    input_mask = Image.open('results/{}/mask.png'.format(formatted_time)).convert('L').resize((h, w))
    image, mask = prepare_img_and_mask(source_image, input_mask, 'cpu')
    outputs = model.run(None, {'image': image.numpy().astype(np.float32), 'mask': mask.numpy().astype(np.float32)})

    output = outputs[0][0].transpose(1, 2, 0).astype(np.uint8)
    path = os.path.join('results/', formatted_time)
    Image.fromarray(output).save(os.path.join(path, 'background.png'))
    return output


def run_openpose(taget):
    apply_openpose = OpenposeDetector()
    input_image = HWC3(taget)
    detected_map, _ = apply_openpose(resize_image(input_image, 512), True)
    return detected_map


def run_finetune(foreground, description):
    assert description != '', 'description is NULL'
    path = os.path.join('results/', formatted_time)
    train_lora(
        name='test.safetensors',
        image=Image.open(os.path.join(path, 'foreground.png')).convert('RGB'),
        prompt=description,
        negative_prompt='',
        model_path='/data/fandongqi/RealVisXL_V4.0',
        save_lora_path=path,
        lora_step=60,
        lora_lr=0.001,
        lora_batch_size=2,
        lora_rank=32,
        no_cfg_probability=1,
        cfg_scale=3.0,
        save_interval=-1,
    )
    return foreground


def run_reposing(pose_image, description):
    assert description != '', 'description is NULL'
    path = os.path.join('results/', formatted_time)

    source_rgba_img = Image.open(os.path.join(path, 'foreground.png')).resize((1024, 1024))
    pose_image = Image.fromarray(pose_image).resize((1024, 1024))
    lora = os.path.join(path, 'test.safetensors')
    negative_prompt = "low quality, bad quality"

    latents_result = []
    def get_latents(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        latents_result.append(latents)
        return callback_kwargs

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
    transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)
    pipe = StableDiffusionXLOnePoseTransPipeline.from_pretrained(base_model, controlnet=controlnet,
                                                                 torch_dtype=torch.float16)

    sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
    sd_origin = pipe.unet.state_dict()
    sd_merged = {}
    for k in sd_origin.keys():
        if k in sd_offset:
            sd_merged[k] = sd_origin[k] + sd_offset[k]
        else:
            sd_merged[k] = sd_origin[k]
    pipe.unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, k

    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # prepare face embedding
    face_info = app.get(cv2.cvtColor(np.array(source_rgba_img), cv2.COLOR_RGB2BGR))
    if face_info == []:
        face_emb = None
        face_flag = False
    else:
        face_emb = face_info[0]['embedding']
        face_flag = True

    lora_scale = {"unet": {
        "down": {"block_1": [0.2, 0.2], "block_2": [0.2, 0.2]},
        "mid": 0.2,
        "up": {"block_0": [0.2, 1.0, 0.2], "block_1": [0.2, 0.2, 0.2]}}}
    pipe.cuda()
    pipe.load_adapters(face_adapter, ip_ckpt)
    generator = torch.Generator('cuda').manual_seed(97)
    pipe.load_lora_weights(lora, 'lora')
    pipe.set_ip_adapter_scale()
    pipe.set_adapters('lora', lora_scale)

    # prepare image embedding
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to('cuda', dtype=torch.float16)
    clip_image_processor = CLIPImageProcessor()
    clip_image = clip_image_processor(images=[source_rgba_img], return_tensors="pt").pixel_values
    clip_image_embeds = image_encoder(clip_image.to('cuda', dtype=torch.float16)).image_embeds
    image_prompt_embeds = pipe.image_proj_model(clip_image_embeds)
    uncond_image_prompt_embeds = pipe.image_proj_model(torch.zeros_like(clip_image_embeds))
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, 1, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * 1, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, 1, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * 1, seq_len, -1)

    # prepare text embedding
    with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            description,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

    # generate image
    image = pipe(
        image_embeds=face_emb,
        face_flag=face_flag,
        image=pose_image,
        control_guidance_start=0,
        control_guidance_end=1,
        prompt_embeds=prompt_embeds,
        num_inference_steps=25,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        generator=generator,
        num_images_per_prompt=1,
        callback_on_step_end=get_latents,
    ).images
    images = torch.from_numpy(np.array(image).transpose((0, 3, 1, 2)).astype(np.float32)) / 255.

    memory_management.load_models_to_gpu([transparent_decoder])
    latents = latents_result[-1].to(dtype=transparent_decoder.model.dtype, device=transparent_decoder.model.device)
    images = images.to(dtype=transparent_decoder.model.dtype, device=transparent_decoder.model.device)
    result_list, vis_list = transparent_decoder(images, latents)

    mask, rgb = run_segmentation_dino(result_list[0][:, :, :3], mode='rgba')
    cv2.imwrite(os.path.join(path, 'mask.png'), mask)
    cv2.imwrite(os.path.join(path, 'rgb.png'), rgb)
    return result_list[0]


def run_refining(description):
    class IPAdapterXL(IPAdapter):
        """SDXL"""

        def generate(
                self,
                pil_image,
                prompt,
                negative_prompt,
                num_images_per_prompt,
                scale,
                num_samples,
        ):
            self.set_scale()

            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            with torch.inference_mode():
                prompt_embeds, pooled_prompt_embeds = self.pipe.encode_cropped_prompt_77tokens(prompt)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_cropped_prompt_77tokens(
                    negative_prompt)

                prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    path = os.path.join('results/', formatted_time)
    assert description != '', 'description is NULL'

    source_rgba_img = Image.open(os.path.join(path, 'foreground.png')).resize((1024, 1024))
    input_image = Image.open(os.path.join(path, 'rgba.png')).resize((1024, 1024))
    lora = os.path.join(path, 'test.safetensors')
    negative_prompt = "low quality, bad quality"

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        base_model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # layerdiffuse injection
    sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
    sd_origin = unet.state_dict()
    sd_merged = {}
    for k in sd_origin.keys():
        if k in sd_offset:
            sd_merged[k] = sd_origin[k] + sd_offset[k]
        else:
            sd_merged[k] = sd_origin[k]
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, k

    transparent_encoder = TransparentVAEEncoder(path_ld_diffusers_sdxl_vae_transparent_encoder)
    transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)

    # Pipelines
    pipeline = KDiffusionStableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=None,
    )
    pipeline.k_model_init(unet)

    lora_scale = {"unet": {
        "down": {"block_1": [0.0, 0.0], "block_2": [0.0, 0.0]},
        "mid": 0.0,
        "up": {"block_0": [0.0, 1.0, 0.0], "block_1": [0.0, 0.0, 0.0]}},
    }
    pipeline.load_lora_weights(lora, 'lora')
    pipeline.set_adapters('lora', lora_scale)

    ip_model = IPAdapterXL(pipeline, image_encoder_path, ip_ckpt, memory_management.gpu)
    initial_latent = [np.array(input_image)]

    with torch.inference_mode():
        guidance_scale = 7.0

        rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)
        memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

        positive_cond, negative_cond, positive_pooler, negative_pooler = ip_model.generate(
            pil_image=source_rgba_img,
            prompt=description,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            scale=1,
            num_samples=1,
        )

        memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])
        initial_latent = transparent_encoder(vae, initial_latent) * vae.config.scaling_factor
        memory_management.load_models_to_gpu([unet])
        initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

        latents = pipeline(
            initial_latent=initial_latent,
            strength=0.65,
            num_inference_steps=25,
            batch_size=1,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=guidance_scale,
        ).images

        memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        result_list, vis_list = transparent_decoder(vae, latents)

        mask, rgb = run_segmentation_dino(result_list[0][:, :, :3], mode='rgb')
        cv2.imwrite(os.path.join(path, 'mask_refined.png'), mask)
        cv2.imwrite(os.path.join(path, 'rgb_refined.png'), rgb)

        return result_list[0]


def run_change_back(background):
    path = os.path.join('results/', formatted_time)
    mask = cv2.imread(os.path.join(path, 'mask.png'))
    mask = (mask == 255).astype(np.uint8)
    rgb = cv2.imread(os.path.join(path, 'rgb.png'))
    background = cv2.resize(background, (1024, 1024))
    Image.fromarray(rgb + background * (1 - mask)).save(os.path.join(path, 'reposed.png'))

    if os.path.isfile(os.path.join(path, 'mask_refined.png')):
        mask_refined = cv2.imread(os.path.join(path, 'mask_refined.png'))
        mask_refined = (mask_refined == 255).astype(np.uint8)
        rgb_refined = cv2.imread(os.path.join(path, 'rgb_refined.png'))
        Image.fromarray(rgb + background * (1 - mask_refined)).save(os.path.join(path, 'refined.png'))
        return rgb + background * (1 - mask), rgb_refined + background * (1 - mask_refined)
    else:
        return rgb + background, None


def run_gemini(image):
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    genai.configure(api_key=os.environ['API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-pro', safety_settings=safety_settings)
    with open('MLLM/template.txt', 'r') as f:
        template = f.readlines()
    prompt = f"{' '.join(template)}"
    response = model.generate_content([prompt, Image.fromarray(image)])
    return response.text


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """<p style="text-align: center; font-size: 32px; front-weight: bold;">OnePoseTrans</p>""")
    with gr.Row():
        gr.Markdown("""<p style="text-align: center; font-size: 25px; front-weight: bold">🙎➡️🙆💁‍♀️🙋</p>""")

    with gr.Accordion(label='', open=True):
        with gr.Row():
            selected_points = gr.State([])
            original_image = gr.State(value=None)
            target = gr.State(value=None)
            pose = gr.State(value=None)
            description = gr.State(value=None)

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Input</p>""")
                input_image = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Foreground</p>""")
                foreground = gr.Image(type="pil", height=length_gradio, width=width_gradio, image_mode='RGBA')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Background</p>""")
                background = gr.Image(type="pil", height=length_gradio, width=width_gradio, image_mode='RGBA')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Inpainted</p>""")
                inpainted = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')

    with gr.Row():
        button_seg_dot = gr.Button('Run Segmentation (Manual dotting)')
        button_seg_dino = gr.Button('Run Segmentation (GroundingDINO)')
        button_inpaint = gr.Button('Run Inpainting')
        button_finetune = gr.Button('Run Finetuning')

    with gr.Row():
        button_openpose = gr.Button('Run Openpose')
        button_repose = gr.Button('Run Pose Transfer')
        button_refine = gr.Button('Run Refining')
        button_change_back = gr.Button('Run Changing Background')

    with gr.Row():
        button_gemini = gr.Button('Run Gemini', scale=1)
        description = gr.Textbox(label='Description', scale=5)

    with gr.Accordion(label='', open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Target</p>""")
                target = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Pose</p>""")
                pose = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Reposed</p>""")
                reposed = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 15px">Refined</p>""")
                refined = gr.Image(type="numpy", height=length_gradio, width=width_gradio, image_mode='RGB')

    def get_point(img, sel_pix, evt: gr.SelectData):
        sel_pix.append((evt.index, 1))
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return img if isinstance(img, np.ndarray) else np.array(img)
    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )

    def store_img(img):
        return img, []
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )


### running
    button_seg_dot.click(run_segmentation_dot, inputs=[original_image, selected_points], outputs=[foreground, background])
    button_seg_dino.click(run_segmentation_dino, inputs=[original_image], outputs=[foreground, background])
    button_inpaint.click(run_inpainting, inputs=[original_image], outputs=[inpainted])
    button_openpose.click(run_openpose, inputs=[target], outputs=[pose])
    button_finetune.click(run_finetune, inputs=[foreground, description], outputs=foreground)
    button_repose.click(run_reposing, inputs=[pose, description], outputs=[reposed])
    button_change_back.click(run_change_back, inputs=[inpainted], outputs=[reposed, refined])
    button_refine.click(run_refining, inputs=[description], outputs=[refined])
    button_gemini.click(run_gemini, inputs=[original_image], outputs=[description])


if __name__ == "__main__":
    demo.launch(debug=True, server_name='')
