import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import csv
import cv2
import numpy as np
import pandas as pd
import safetensors.torch as sf

from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from diffusers import ControlNetModel
from PIL import Image
from lib_layerdiffuse.vae import TransparentVAEDecoder
from utils.pose_utils import draw_pose_from_cords, load_pose_cords_from_strings
from utils import memory_management
from insightface.app import FaceAnalysis
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from pipeline_stable_diffusion_xl_oneposetrans import StableDiffusionXLOnePoseTransPipeline


def build_pose_img(annotation_file, img_path):
    string = annotation_file.loc[os.path.basename(img_path)]
    array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
    pose_img = draw_pose_from_cords(array, (256, 256), (256, 176))
    return Image.fromarray(pose_img).resize((1024, 1024))


def build_pose_img_openpose(img_path):
    apply_openpose = OpenposeDetector()
    input_image = cv2.imread(img_path)
    input_image = HWC3(input_image)
    detected_map, _ = apply_openpose(resize_image(input_image, 1024), True)
    return detected_map


# get source and target paris
source_imgs = []
target_imgs = []
with open('files/fasion-resize-pairs-test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in list(reader)[1:]:
        source_imgs.append(row[0])
        target_imgs.append(row[1])
annotation_file = pd.read_csv("files/fasion-resize-annotation-test.csv", sep=':')
annotation_file = annotation_file.set_index('name')


# get discrete descriptions
negative_prompt = "low quality, bad quality"
with open('files/fashion_descriptions.txt', 'r') as f:
    all_lines = f.readlines()
descriptions = dict()
for line in all_lines:
    line = line.rstrip()
    _, name, description = line.split('/')[0], line.split('/')[1], line.split('/')[2]
    descriptions.update({name:description})


# get deepfashion test list
check_list = []
with open('files/fashion_test_pairs.txt', 'r') as f:
    all_lines = f.readlines()
for line in all_lines:
    line = line.rstrip()
    _, name1, name2 = line.split('/')[0], line.split('/')[1], line.split('/')[2]
    check_list.append(name1 + name2)


# datasets
rgba_path = '/your/path/to/save/results/images_rgba'
lora_path = '/your/path/to/save/results/lora'
real_path = '/your/path/to/save/results/images'

# InstantID
face_adapter = 'models/InstantID/ip-adapter.bin'

# IP-Adapter
ip_adapter = 'models/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin'
image_encoder_path = "models/IP-Adapter/models/image_encoder"

# SDXL
base_model = 'models/RealVisXL_V4.0'

# ControlNet
controlnet_model = 'models/controlnet-openpose-sdxl-1.0'

# LayerDiffuse
path_ld_diffusers_sdxl_attn = 'models/LayerDiffuse/ld_diffusers_sdxl_attn.safetensors'
path_ld_diffusers_sdxl_vae_transparent_decoder = 'models/LayerDiffuse/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'

for source_img, target_img in zip(source_imgs, target_imgs):

    if source_img + target_img not in check_list: continue

    source_rgba_img = Image.open(rgba_path + source_img.replace('jpg', 'png')).resize((1024, 1024))
    pose_image = build_pose_img(annotation_file, real_path + target_img)
    # pose_image = build_pose_img_openpose(source_path + to_img)

    prompt = descriptions[source_img]
    lora = lora_path + source_img.replace('jpg', 'safetensors')
    assert os.path.isfile(lora), "{} not exist".format(lora)

    # latents are used for transparent decoder
    latents_result = []
    def get_latents(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        latents_result.append(latents)
        return callback_kwargs

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
    transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)
    pipe = StableDiffusionXLOnePoseTransPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)

    # layerdiffuse injection
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

    # prepare 'antelopev2' under ./models
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # prepare face embedding
    # sometimes face may not be detected in foreground rgba img, try whole source img
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
    pipe.load_adapters(face_adapter, ip_adapter)
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
            prompt,
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

    # decode for transparent
    memory_management.load_models_to_gpu([transparent_decoder])
    latents = latents_result[-1].to(dtype=transparent_decoder.model.dtype, device=transparent_decoder.model.device)
    images = images.to(dtype=transparent_decoder.model.dtype, device=transparent_decoder.model.device)
    result_list, vis_list = transparent_decoder(images, latents)

    # saving
    Image.fromarray(result_list[0]).save('/your/path/to/save/results/reposed_result/{}_2_{}.png'.format(
        source_img.split('.')[0], target_img.split('.')[0]))
