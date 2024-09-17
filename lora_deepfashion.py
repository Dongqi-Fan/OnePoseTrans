# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
)
from peft.utils import get_peft_model_state_dict

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def train_lora(
    name,
    image,
    prompt,
    negative_prompt,
    model_path,
    save_lora_path,
    lora_step,
    lora_lr,
    lora_batch_size,
    lora_rank,
    no_cfg_probability,
    cfg_scale,
    save_interval=-1
):
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
    set_seed(0)

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                 torch_dtype=torch.float16, variant="fp16")
    text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder_2",
                                                   torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16")


    # set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    text_encoder_2.to(device, dtype=torch.float16)

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    cast_training_params(unet, dtype=torch.float32)

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_step,
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    unet_lora_layers = accelerator.prepare_model(unet)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    def prompt_encode(prompt, tokenizers, text_encoders, device):
        pooled_prompt_embeds = None
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # Only last pooler_output is needed
            pooled_prompt_embeds = prompt_embeds.pooler_output

            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.repeat(lora_batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(lora_batch_size, 1)

        add_time_ids = list((image.size[0], image.size[1]) + (0, 0) + (image.size[0], image.size[1]))
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(lora_batch_size, 1).to(device)
        return prompt_embeds, pooled_prompt_embeds, add_time_ids


    # initialize text embeddings
    with torch.no_grad():
        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]

        positive_prompt_embeds, positive_pooled_prompt_embeds, add_time_ids = \
            prompt_encode(prompt, tokenizers, text_encoders, device)
        negative_prompt_embeds, negative_pooled_prompt_embeds, _ = \
            prompt_encode(negative_prompt, tokenizers, text_encoders, device)

        sampler_kwargs = dict(
            positive=dict(
                encoder_hidden_states=positive_prompt_embeds,
                added_cond_kwargs={"text_embeds": positive_pooled_prompt_embeds, "time_ids": add_time_ids}),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_time_ids})
        )

    # initialize latent distribution
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    for step in tqdm(range(lora_step), desc="training LoRA"):
        unet.train()
        image_batch = []
        for _ in range(lora_batch_size):
            image_transformed = image_transforms(image).to(device, dtype=torch.float16)
            image_transformed = image_transformed.unsqueeze(dim=0)
            image_batch.append(image_transformed)

        # repeat the image_transformed to enable multi-batch training
        image_batch = torch.cat(image_batch, dim=0)

        latents_dist = vae.encode(image_batch).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        random_number = random.random()
        if random_number > no_cfg_probability:
            eps_positive = unet(noisy_model_input, timesteps, **sampler_kwargs['positive']).sample
            eps_negative = unet(noisy_model_input, timesteps, **sampler_kwargs['negative']).sample
            model_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        else:
            model_pred = unet(noisy_model_input, timesteps, **sampler_kwargs['positive']).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if save_interval > 0 and (step + 1) % save_interval == 0:
            save_lora_path_intermediate = os.path.join(save_lora_path, str(step+1))
            if not os.path.isdir(save_lora_path_intermediate):
                os.mkdir(save_lora_path_intermediate)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_lora_path_intermediate,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=save_lora_path,
        unet_lora_layers=unet_lora_state_dict,
        weight_name=name,
    )


if __name__ == "__main__":
    images_path = 'your/path/results/images'
    with open('files/fashion_descriptions.txt', 'r') as f:
        all_lines = f.readlines()
    all_index = []
    all_name = []
    all_description = []
    for line in all_lines:
        line = line.rstrip()
        index, img_name, description = line.split('/')[0], line.split('/')[1], line.split('/')[2]
        all_index.append(index)
        all_name.append(img_name)
        all_description.append(description)

    for index, name, prompt in zip(all_index, all_name, all_description):
        train_lora(
            name=name.split('.')[0] + '.safetensors',
            image=Image.open(images_path + name.split('.')[0] + '.png').convert('RGB'),
            prompt=prompt,
            negative_prompt='',
            model_path='models/RealVisXL_V4.0',
            save_lora_path='/your/path/to/save/results/lora',
            lora_step=60,
            lora_lr=0.001,
            lora_batch_size=2,
            lora_rank=32,
            no_cfg_probability=1,
            cfg_scale=3.0,
            save_interval=-1,
        )