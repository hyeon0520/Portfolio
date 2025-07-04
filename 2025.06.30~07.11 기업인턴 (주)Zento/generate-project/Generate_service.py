from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Generate import generate_image
from MMT import contains_korean, translate_ko_to_en
import uuid

# FastAPI 객체 생성
app = FastAPI()

# 요청 스키마 정의
class PromptRequest(BaseModel):
    user_input: str
    prompt: str

@app.post("/generate")
def generate(request: PromptRequest):
    model = request.user_input
    prompt = request.prompt
    if contains_korean(request.prompt):
        prompt = translate_ko_to_en(request.prompt)
    
    image = generate_image(model, prompt)

    file_path = f"{uuid.uuid4().hex}.png"
    
    image.save(file_path)
    
    return FileResponse(path=file_path, media_type="image/png", filename="output.png")