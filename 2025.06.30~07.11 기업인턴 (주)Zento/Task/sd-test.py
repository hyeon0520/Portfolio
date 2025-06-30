import torch
from diffusers import StableDiffusionPipeline

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A group of young people working together as corporate interns, smiling and collaborating in a modern office environment"
image = pipe(prompt).images[0]
    
image.save("yes, we are.png")