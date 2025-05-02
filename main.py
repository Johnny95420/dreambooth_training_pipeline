# %%
import random
import copy
import functools
import logging
import os
import shutil
import diffusers
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DreamBoothDataset, collate_fn
from omegaconf import OmegaConf

# %%
logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

config = OmegaConf.load("config.yaml")
tracker_name = "orii_plush_doll"
lr_warmup_steps = config["training_config"]["warmup_steps"]
max_train_steps = config["training_config"]["max_steps"]
pretrained_model_name_or_path = config["model_config"]["model_id"]
gradient_accumulation_steps = config["training_config"]["gradient_accumulation_steps"]
learning_rate = config["training_config"]["lr"]
prior_loss_weight = config["training_config"]["prior_loss_weight"]
batch_size = config["training_config"]["batch_size"]
output_dir = config["training_config"]["ckpt_dir"]
logging_dir = config["training_config"]["logging_dir"]
upcast_before_saving = config["training_config"]["upcast_before_saving"]
special_token = config["data_config"]["special_word"]
class_prompt = config["data_config"]["class_prompt"]
instance_prompt = config["data_config"]["instance_prompt"].format(
    special_word=special_token
)


# %%
def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    if not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        transformer_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                if upcast_before_saving:
                    model = model.to(torch.float32)
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

        StableDiffusion3Pipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )


def load_model_hook(models, input_dir):
    transformer_ = None
    if not accelerator.distributed_type == DistributedType.DEEPSPEED:
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

    else:
        transformer_ = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer"
        )
        transformer_.add_adapter(transformer_lora_config)
    lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

    transformer_state_dict = {
        f"{k.replace('transformer.', '')}": v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(
        transformer_, transformer_state_dict, adapter_name="default"
    )
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )


def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def validation_generation(
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    global_step,
    logging_dir,
):
    if (
        accelerator.is_main_process
        or accelerator.distributed_type == DistributedType.DEEPSPEED
    ):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
        )
        with torch.autocast("cuda"):
            image = pipe(
                guidance_scale=4,
                num_inference_steps=25,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=768,
                height=768,
            )
        del pipe
        image.images[0].save(f"{logging_dir}/img_{global_step}.jpg")
        torch.cuda.empty_cache()
        return image.images[0]


def training_step(
    transformer,
    noise_scheduler_copy,
    vae_config_shift_factor,
    vae_config_scaling_factor,
):
    idx = random.randint(0, len(latents_cache) - 1)
    model_input = latents_cache[idx].sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=torch.bfloat16)
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=bsz,
        logit_mean=0.0,
        logit_std=1.0,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    # Add noise according to flow matching.
    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    # Predict the noise residual
    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]
    model_pred = model_pred * (-sigmas) + noisy_model_input

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme="logit_normal", sigmas=sigmas
    )
    target = model_input
    weighting_pred, weighting_prior = torch.chunk(weighting, 2, dim=0)

    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    prior_loss = torch.mean(
        (
            weighting_prior.float()
            * (model_pred_prior.float() - target_prior.float()) ** 2
        ).reshape(target_prior.shape[0], -1),
        1,
    )
    prior_loss = prior_loss.mean()

    instance_loss = torch.mean(
        (weighting_pred.float() * (model_pred.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        ),
        1,
    )
    instance_loss = instance_loss.mean()
    loss = instance_loss + prior_loss_weight * prior_loss
    return instance_loss, prior_loss, loss


def compute_grad_norm(transformer):
    total_norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), 2)
                for p in accelerator.unwrap_model(transformer).parameters()
                if p.grad is not None
            ]
        )
    )
    return total_norm


def save_ckpt(accelerator, global_step):
    if (
        accelerator.is_main_process
        or accelerator.distributed_type == DistributedType.DEEPSPEED
    ):
        if checkpoints_total_limit is not None:
            checkpoints = os.listdir(output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= checkpoints_total_limit:
                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


# %%
if __name__ == "__main__":
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers(tracker_name)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    # %%
    vae_cache = torch.load("/dreambooth/latent_cache.pt")
    latents_cache, vae_config_shift_factor, vae_config_scaling_factor = (
        vae_cache["latent_cache"],
        vae_cache["vae_config_shift_factor"],
        vae_cache["vae_config_scaling_factor"],
    )

    text_embeds = torch.load("text_embeds.pt")
    prompt_embeds, pooled_prompt_embeds = (
        text_embeds["prompt_embeds"],
        text_embeds["pooled_prompt_embeds"],
    )

    # %%
    # * get transforemr
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    target_blocks = [
        block
        for block in [
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
        ]
    ]
    target_modules = [
        f"transformer_blocks.{block}.{module}"
        for block in target_blocks
        for module in target_modules
    ]
    transformer_lora_config = LoraConfig(
        r=64, lora_alpha=64, target_modules=target_modules
    )
    transformer.add_adapter(transformer_lora_config)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=1,
    )
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )
    # %%
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # %%
    global_step = 0
    path = "/dreambooth/output/checkpoint-700"
    accelerator.load_state(path)
    global_step = int(path.split("-")[1])
    initial_global_step = global_step
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    checkpointing_steps = 100
    validation_steps = 20
    checkpoints_total_limit = 5

    flag = True
    models_to_accumulate = [transformer]
    while flag:
        for step in range(len(latents_cache)):
            with accelerator.accumulate(models_to_accumulate):
                instance_loss, prior_loss, loss = training_step(
                    transformer,
                    noise_scheduler_copy,
                    vae_config_shift_factor,
                    vae_config_scaling_factor,
                )
                accelerator.backward(loss)
                grad_norm = compute_grad_norm(transformer)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, 3)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {
                "instance_loss": instance_loss.detach().item(),
                "prior_loss": prior_loss.detach().item(),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "gradient_norm": grad_norm.item(),
            }
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % checkpointing_steps == 0:
                    save_ckpt(accelerator, global_step)
                if global_step % validation_steps == 0:
                    validation_generation(
                        prompt_embeds[[0]],
                        pooled_prompt_embeds[[0]],
                        text_embeds["negative_prompt_embeds"][[0]],
                        text_embeds["negative_pooled_prompt_embeds"][[0]],
                        global_step,
                        logging_dir,
                    )
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            if global_step >= max_train_steps:
                flag = False
                break
    # %%
