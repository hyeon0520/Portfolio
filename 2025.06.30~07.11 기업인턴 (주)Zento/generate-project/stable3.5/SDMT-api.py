from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import MarianMTModel, MarianTokenizer
import torch
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from transformers import MarianMTModel, MarianTokenizer
import torch
import uuid
import os
from MMT import contains_korean
from MMT import translate_ko_to_en

# FastAPI 객체 생성
app = FastAPI()

# 생성 모델 정의
model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

# 한글 포함 여부 판별 함수
def contains_korean(text):
    return bool(re.search(r"[가-힣]", text))

# 요청 스키마 정의
class PromptRequest(BaseModel):
    prompt: str

@app.post("/SD3.5")
def generate_image(data: PromptRequest):
    prompt = data.prompt
    if contains_korean(data.prompt):
        prompt = translate_ko_to_en(data.prompt)

    image = pipeline(prompt,
                 negative_prompt="blurry, low quality, low resolution",
                 num_inference_steps=50,
                 guidance_scale=5.5,
                 height=512,
                 width=512,
                 num_images_per_prompt=1,
                 max_sequence_length=256
                 ).images[0]
    
    # 이미지 저장
    file_path = f"{uuid.uuid4().hex}.png"
    image.save(file_path)
    
    return FileResponse(path=file_path, media_type="image/png", filename="output.png")