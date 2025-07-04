from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch


#모델 파이프라인 호출 함수
def load_SD3():
        model_id = "stabilityai/stable-diffusion-3.5-medium"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16
        )

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            transformer=model_nf4,
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()

        return pipeline

#이미지 생성 함수
def generate_image_SD3(prompt: str):
        pipeline = load_SD3()
        image = pipeline(
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=5.5,
        max_sequence_length=512,
        ).images[0]

        return image