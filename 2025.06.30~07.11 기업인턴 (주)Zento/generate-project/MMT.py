from transformers import MarianMTModel, MarianTokenizer
import re

# 모델 정의
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 한글 포함 여부 판별 함수
def contains_korean(text):
    return bool(re.search(r"[가-힣]", text))

# 번역 함수 정의
def translate_ko_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text

# # 토큰화
# inputs = tokenizer(src_text, return_tensors="pt", padding=True)
# # 번역
# translated = model.generate(**inputs)
# # 디코딩
# translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

# print(translated_text)