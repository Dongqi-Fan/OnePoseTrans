import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from utils import memory_management
from utils.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from ip_adapter import IPAdapter
from typing import List


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


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
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_cropped_prompt_77tokens(negative_prompt)

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


# datasets
rgba_path = '/your/path/to/save/results/images'
lora_path = '/your/path/to/save/results/lora'
input_path = '/your/path/to/save/results/reposed_rgba'

# IP-Adapter
ip_adapter = 'models/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin'
image_encoder_path = "models/IP-Adapter/models/image_encoder"

# SDXL
base_model = 'models/RealVisXL_V4.0'

# LayerDiffuse
path_ld_diffusers_sdxl_attn = 'models/LayerDiffuse/ld_diffusers_sdxl_attn.safetensors'
path_ld_diffusers_sdxl_vae_transparent_decoder = 'models/LayerDiffuse/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
path_ld_diffusers_sdxl_vae_transparent_encoder = 'models/LayerDiffuse/ld_diffusers_sdxl_vae_transparent_encoder.safetensors'

# get discrete descriptions
negative_prompt = "low quality, bad quality"
with open('files/fashion_descriptions.txt', 'r') as f:
    all_lines = f.readlines()
descriptions = dict()
for line in all_lines:
    line = line.rstrip()
    _, name, description = line.split('/')[0], line.split('/')[1], line.split('/')[2]
    descriptions.update({name:description})


for img in os.listdir(input_path):
    from_name = img.split('_2_')[0] + '.png'
    input_img = Image.open(input_path + img).resize((1024, 1024))
    ref_img = Image.open(rgba_path + from_name).resize((1024, 1024))
    
    prompt = descriptions[from_name.replace('png', 'jpg')]
    lora = lora_path + from_name.replace('png', 'safetensors')
    assert os.path.isfile(lora), "{} not exist".format(lora)

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
    keys = sd_origin.keys()
    sd_merged = {}
    for k in sd_origin.keys():
        if k in sd_offset:
            sd_merged[k] = sd_origin[k] + sd_offset[k]
        else:
            sd_merged[k] = sd_origin[k]
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, keys, k

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

    ip_model = IPAdapterXL(pipeline, image_encoder_path, ip_adapter, memory_management.gpu)
    initial_latent = [np.array(input_img)]

    with torch.inference_mode():
        guidance_scale = 7.0

        rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)
        memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

        positive_cond, negative_cond, positive_pooler, negative_pooler = ip_model.generate(
            pil_image=ref_img,
            prompt=prompt,
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
        Image.fromarray(result_list[0]).save("test/{}".format(img), format='PNG')
