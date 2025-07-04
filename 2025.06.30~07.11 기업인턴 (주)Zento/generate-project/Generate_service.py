from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from Generate import generate_image
from MMT import contains_korean, translate_ko_to_en
import uuid
import os

# FastAPI 객체 생성
app = FastAPI()

# 모델 정보 추가
available_models = {
    "SD3": {"name": "Stable-Diffuison-v3.5 Medium", "Time": "15 ~ 20"}
}

# 요청 스키마 정의
class PromptRequest(BaseModel):
    model: str
    prompt: str

# 모델 목록 반환
@app.get("/models")
def get_models():
    return JSONResponse(content=available_models)

# 이미지 생성
@app.post("/generate")
def generate(request: PromptRequest):
    model = request.model
    prompt = request.prompt

    if model not in available_models:
        raise HTTPException(status_code=400, detail="Invalid model selected.")

    if contains_korean(request.prompt):
        prompt = translate_ko_to_en(request.prompt)
    
    image = generate_image(model, prompt)
    image_id = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join("generate_images", image_id)
    
    # 저장 디렉토리 없으면 생성
    os.makedirs("generate_images", exist_ok=True)
    image.save(image_path)

    return {
        "success": True,
        "image_id": image_id,
        "message": "Image generated successfully."
    }

# 이미지 반환
@app.get("/image/{image_id}")
def get_image(image_id: str):
    image_path = os.path.join("generate_images", image_id)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(path=image_path, media_type="image/png", filename=image_id)