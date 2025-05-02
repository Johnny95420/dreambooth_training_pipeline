# %%
import functools
import torch
from diffusers import (
    AutoencoderKL,
)
from diffusers.training_utils import (
    free_memory,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from dataset import DreamBoothDataset, collate_fn
from omegaconf import OmegaConf

# %%
config = OmegaConf.load("config.yaml")
batch_size = config["training_config"]["batch_size"]
pretrained_model_name_or_path = config["model_config"]["model_id"]
special_token = config["data_config"]["special_word"]
config["data_config"]["instance_prompt"] = config["data_config"][
    "instance_prompt"
].format(special_word=special_token)


# %%
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    _, dim = pooled_prompt_embeds.shape
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, dim
    )

    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(
        zip(clip_tokenizers, clip_text_encoders)
    ):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


# %%
if __name__ == "__main__":
    dataset = DreamBoothDataset(
        instance_data_root=config["data_config"]["instance_img_dir"],
        class_data_root=config["data_config"]["class_img_dir"],
        class_num=200,
        instance_prompt="",
        class_prompt="",
        size=config["training_config"]["img_size"],
        center_crop=False,
        resolution=config["training_config"]["center_crop_res"],
        repeats=1,
    )
    collate_fn = functools.partial(collate_fn, with_prior_preservation=True)
    train_dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=batch_size,
    )
    # %%
    # * get vae and prefix image latent
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
    ).to("cuda", dtype=torch.float32)
    vae.requires_grad_(False)
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    latents_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(
                "cuda", dtype=torch.float32
            )
            latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
    del vae
    free_memory()
    torch.save(
        {
            "latent_cache": latents_cache,
            "vae_config_shift_factor": vae_config_shift_factor,
            "vae_config_scaling_factor": vae_config_scaling_factor,
        },
        "latent_cache.pt",
    )
    # %%
    # Load the tokenizers
    # * get prefix pool text embedding  and text seq embeddings
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_3",
    )
    # * get text encoder 1,2,3
    text_encoder_one, text_encoder_two, text_encoder_three = (
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).to(torch.bfloat16),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).to(torch.bfloat16),
        T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ),
    )
    with torch.no_grad():
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = encode_prompt(
            [text_encoder_one, text_encoder_two, text_encoder_three],
            [tokenizer_one, tokenizer_two, tokenizer_three],
            prompt=config["data_config"]["instance_prompt"],
            max_sequence_length=77,
            device="cuda",
            num_images_per_prompt=batch_size,
        )
        class_prompt_hidden_states, class_pooled_prompt_embeds = encode_prompt(
            [text_encoder_one, text_encoder_two, text_encoder_three],
            [tokenizer_one, tokenizer_two, tokenizer_three],
            prompt=config["data_config"]["class_prompt"],
            max_sequence_length=77,
            device="cuda",
            num_images_per_prompt=batch_size,
        )
        negative_prompt_hidden_states, negative_pooled_prompt_embeds = encode_prompt(
            [text_encoder_one, text_encoder_two, text_encoder_three],
            [tokenizer_one, tokenizer_two, tokenizer_three],
            prompt="",
            max_sequence_length=77,
            device="cuda",
            num_images_per_prompt=batch_size,
        )
    prompt_embeds = torch.cat(
        [instance_prompt_hidden_states, class_prompt_hidden_states], dim=0
    )
    pooled_prompt_embeds = torch.cat(
        [instance_pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0
    )

    del text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()
    torch.save(
        {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_hidden_states,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        },
        "text_embeds.pt",
    )
# %%
