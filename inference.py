# %%
from PIL import Image
import torch
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

# StableDiffusion3Pipeline.lora_state_dict("/dreambooth/output/checkpoint-1000")
pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-large"


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
    device_map="cuda",
)
# %%
pipe = StableDiffusion3Pipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.bfloat16)
pipe.enable_xformers_memory_efficient_attention()
pipe.load_lora_weights(
    "/dreambooth/output/checkpoint-1000",
    weight_name="pytorch_lora_weights.safetensors",
)

pipe.load_ip_adapter(
    "/dreambooth/ip_adapter",
    "ip-adapter.bin",
    image_encoder_folder="google/siglip-so400m-patch14-384",
)
# %%
ref_img = Image.open("./ip_adapter/assets/7.jpg").convert("RGB")
# pipe.set_adapters(["default_0"], adapter_weights=[1.0])
prompt = """a orii plush animal style piglet rides a vintage Vespa"""
pipe.scheduler.set_shift(10)
image = pipe(
    prompt,
    negative_prompt="a plush animal",
    guidance_scale=7,
    num_inference_steps=40,
    width=1024,
    height=1024,
    ip_adapter_image=ref_img,
    skip_guidance_layers=[12, 13, 14],
)
torch.cuda.empty_cache()
image.images[0]
# %%
