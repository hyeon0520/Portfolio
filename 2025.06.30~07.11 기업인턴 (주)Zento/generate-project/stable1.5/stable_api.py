from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import torch
import uuid
import os

app = FastAPI()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# 요청 데이터 스키마 정의

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: PromptRequest):
    prompt = data.prompt
    image = pipe(prompt).images[0]
    
    # 이미지 저장
    file_path = f"{uuid.uuid4().hex}.png"
    image.save(file_path)
    
    return FileResponse(path=file_path, media_type="image/png", filename="output.png")