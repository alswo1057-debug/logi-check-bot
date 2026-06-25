import os
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from common import (
    MODEL_NAME,
    open_image_fixed_from_bytes,
    check_by_auto_classification
)


load_dotenv()

app = FastAPI(title="Logi Check API")


@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Logi Check API is running",
        "model": MODEL_NAME,
        "endpoint": "/check",
        "method": "POST"
    }


@app.post("/check")
async def check(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),
    image5: Optional[UploadFile] = File(None),
    image6: Optional[UploadFile] = File(None),
    image7: Optional[UploadFile] = File(None),
    image8: Optional[UploadFile] = File(None),
    image9: Optional[UploadFile] = File(None),
    image10: Optional[UploadFile] = File(None),
):
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "OPENAI_API_KEY를 찾을 수 없습니다."
                }
            )

        files = [
            image1, image2, image3, image4, image5,
            image6, image7, image8, image9, image10
        ]

        files = [f for f in files if f is not None]

        if len(files) < 2:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "사진은 최소 2장 이상 필요합니다. 피킹리스트 1장 + 상품라벨 1장 이상 업로드해주세요."
                }
            )

        uploaded_files = []

        for idx, file in enumerate(files):
            file_bytes = await file.read()

            if not file_bytes:
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": f"{file.filename or 'image_' + str(idx + 1)} 파일이 비어 있습니다."
                    }
                )

            image = open_image_fixed_from_bytes(file_bytes)

            uploaded_files.append({
                "filename": file.filename or f"image_{idx + 1}.jpg",
                "image": image
            })

        result = check_by_auto_classification(uploaded_files)

        return {
            "ok": True,
            "model": MODEL_NAME,
            "result": result
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
