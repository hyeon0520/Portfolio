import torch
import random
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download

#SD3 모델 생성
def load_SD3():
        base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        repo_name = "ByteDance/Hyper-SD"
        # Take 8-steps lora as an example
        sd3_ckpt_name = "Hyper-SD3-8steps-CFG-lora.safetensors"
        # Load model, please fill in your access tokens since SD3 repo is a gated model.
        pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, token="")
        pipe.load_lora_weights(hf_hub_download(repo_name, sd3_ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to("cuda", dtype=torch.float16)

        return pipe

#AI 생성 함수
def generate_image_SD3(prompt: str):
    pipe = load_SD3()
    image=pipe(prompt=prompt,
            negative_prompt="blurry, low quality, low resolution",
            width=768,
            num_inference_steps=20,
            guidance_scale=5.5,
            #    generator=generator
            ).images[0]
    
    return image

image = generate_image_SD3("A man and a woman warmly holding hands, smiling at each other, in a sunny park setting.")
image.save("output.png")