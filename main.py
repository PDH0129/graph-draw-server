# main.py

import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from groq_client import handwriting_to_latex
import traceback
from sympy import symbols, integrate, diff, solve
from latex2sympy2 import latex2sympy


# -----------------------
# 기본 설정
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Handwritten Math OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

x, y = symbols("x y")


# -----------------------
# LaTeX 전처리
# -----------------------
def clean_latex(latex: str) -> str:
    latex = latex.replace("$$", "")
    latex = latex.replace("\\,", "")
    latex = latex.strip()
    return latex


# -----------------------
# LaTeX → 그래프용 함수 변환
# -----------------------
def resolve_latex_to_functions(latex: str):
    latex = clean_latex(latex)

    # y= 제거
    if latex.startswith("y="):
        latex = latex[2:]

    functions = []

    # 1️⃣ 적분
    if "\\int" in latex:
        expr = latex.replace("\\int", "").replace("dx", "")
        sym = latex2sympy(expr)
        res = integrate(sym, x)
        functions.append(str(res))
        return functions

    # 2️⃣ 미분
    if "\\frac{d}{dx}" in latex or "\\dfrac{d}{dx}" in latex:
        expr = re.sub(r"\\(d)?frac\{d\}\{dx\}", "", latex)
        sym = latex2sympy(expr)
        res = diff(sym, x)
        functions.append(str(res))
        return functions

    # 3️⃣ 방정식 (y 포함)
    if "=" in latex and "y" in latex:
        left, right = latex.split("=")
        sym_left = latex2sympy(left)
        sym_right = latex2sympy(right)

        sols = solve(sym_left - sym_right, y)
        for sol in sols:
            functions.append(str(sol))
        return functions

    # 4️⃣ 일반 함수 / 상수
    sym = latex2sympy(latex)
    functions.append(str(sym))
    return functions


# -----------------------
# API
# -----------------------
@app.post("/handwriting")
async def handwriting_ocr(image: UploadFile = File(...)):
    if image.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="PNG 또는 JPG만 지원")

    image_path = os.path.join(UPLOAD_DIR, image.filename)

    with open(image_path, "wb") as f:
        f.write(await image.read())

    try:
        # 1️⃣ OCR → LaTeX
        latex = handwriting_to_latex(image_path)

        # 2️⃣ LaTeX → 그래프용 함수들
        functions = resolve_latex_to_functions(latex)

        return JSONResponse({
            "latex": latex,          # 원본 OCR LaTeX
            "functions": functions,  # ⭐ 그래프용 (배열)
            "graphable": True
        })

    except Exception as e:
        print("🔥 HANDWRITING ERROR 🔥")
        traceback.print_exc()   # ⭐ 이게 핵심
        raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "latex": latex if "latex" in locals() else None,
                "graphable": False,
                "error": str(e)
            }
        )
