# %%
import os
import torch
from accelerate import Accelerator
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from huggingface_hub.utils import insecure_hashlib
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omegaconf import OmegaConf

# %%
config = OmegaConf.load("config.yaml")
MODEL_ID = config["model_config"]["model_id"]
CLASS_PROMPT = config["data_config"]["class_prompt"]
NUM_NEW_IMAGES = config["data_config"]["num_class_data"]
SAMPLE_BATCH_SIZE = config["data_config"]["sample_generation_batch_size"]
IMG_DIR = config["data_config"]["class_img_dir"]

os.makedirs(IMG_DIR, exist_ok=True)


class PromptDataset(Dataset):

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


# %%
if __name__ == "__main__":
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    mmdit_nf4 = SD3Transformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        transformer=mmdit_nf4,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.bfloat16)
    # %%
    accelerator = Accelerator(mixed_precision="bf16")
    sample_dataset = PromptDataset(CLASS_PROMPT, NUM_NEW_IMAGES)
    sample_dataloader = DataLoader(sample_dataset, batch_size=SAMPLE_BATCH_SIZE)
    sample_dataloader = accelerator.prepare(sample_dataloader)
    pipe.to(accelerator.device)
    for example in tqdm(
        sample_dataloader,
        desc="Generating class images",
        disable=not accelerator.is_local_main_process,
    ):
        images = pipe(
            example["prompt"],
            num_inference_steps=25,
            guidance_scale=6,
            height=1024,
            width=1024,
        ).images
        for i, image in enumerate(images):
            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = f"{IMG_DIR}/{example['index'][i]}-{hash_image}.jpg"
            image.save(image_filename)
