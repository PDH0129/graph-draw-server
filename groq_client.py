import os
import base64
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ======================
# 여기에 API 키 박아라
# ======================

GROQ_API_KEY = "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"

# ======================
# 설정
# ======================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Handwritten Math OCR API")

# ======================
# Groq 호출
# ======================

def handwriting_to_latex(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "손글씨로 쓰인 수학 수식을 인식해서 "
                            "LaTeX 형식으로만 출력해. "
                            "설명, 문장, 코드블록 절대 쓰지 마."
                            "y={}의 형식으로 출력해"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ======================
# API
# ======================

@app.post("/handwriting")
async def handwriting_ocr(image: UploadFile = File(...)):
    if image.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="PNG/JPG만 가능")

    image_path = os.path.join(UPLOAD_DIR, image.filename)

    with open(image_path, "wb") as f:
        f.write(await image.read())

    try:
        latex = handwriting_to_latex(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "latex": latex.strip()
    })
