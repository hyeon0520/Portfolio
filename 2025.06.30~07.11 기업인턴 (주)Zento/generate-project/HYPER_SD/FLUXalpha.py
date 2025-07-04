import torch
import random
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

#FLUX 모델 생성
def load_FLUX():
        base_model_id = "black-forest-labs/FLUX.1-dev"
        repo_name = "ByteDance/Hyper-SD"
        # Take 8-steps lora as an example
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        # Load model, please fill in your access tokens since FLUX.1-dev repo is a gated model.
        pipe = FluxPipeline.from_pretrained(base_model_id, token="")
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to("cuda", dtype=torch.float16)

        return pipe

#AI 생성 함수
def generate_image_FLUX(prompt: str):
    pipe = load_FLUX()
    image=pipe(prompt=prompt,
            negative_prompt="blurry, low quality, low resolution",
            height=1024,
            width=768,
            num_inference_steps=10,
            guidance_scale=5.5,
            #    generator=generator
            ).images[0]
    
    return image