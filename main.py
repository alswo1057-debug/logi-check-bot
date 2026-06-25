import os
import json
import base64
import traceback
from io import BytesIO
from collections import Counter

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from openai import OpenAI

try:
    import zxingcpp
except Exception:
    zxingcpp = None


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI(title="Logi Check API")


def open_image_fixed(file_bytes: bytes):
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def pil_to_base64(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_json_text(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


def normalize_code(value):
    if value is None:
        return ""
    value = str(value).strip().upper()
    for ch in [" ", "\n", "\r", "\t", "-", "_"]:
        value = value.replace(ch, "")
    return value


def gpt_json(prompt: str, images: list[Image.Image]):
    content = [{"type": "input_text", "text": prompt}]

    for img in images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{pil_to_base64(img)}"
        })

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}]
    )

    return json.loads(clean_json_text(response.output_text))


def classify_images(images_info):
    prompt = """
너는 물류 검수 사진 분류 AI다.

여러 장의 이미지가 순서대로 제공된다.
각 이미지가 아래 중 무엇인지 분류하라.

분류값:
- picklist: 피킹리스트, 출고지시서, 제품 포장 정보 표가 있는 문서 사진
- label: 상품 라벨, 바코드 라벨, 제품 실물 바코드 사진
- unknown: 판단 불가

중요 규칙:
- 피킹리스트는 보통 표, Wave No, 제품 포장 정보, EAN, 포장수량 같은 문구가 있다.
- 상품라벨은 보통 바코드, QR, CODE, SKU, 상품명 라벨이 크게 보인다.
- 반드시 JSON만 출력한다.
- 설명하지 마라.

JSON 형식:
{
  "images":[
    {"index":0, "type":"picklist", "reason":"제품 포장 정보 표가 보임"},
    {"index":1, "type":"label", "reason":"바코드 라벨 사진"}
  ]
}
"""

    images = [x["image"] for x in images_info]
    return gpt_json(prompt, images)


def read_picklist_with_gpt(image: Image.Image):
    prompt = """
너는 물류 피킹리스트 판독 AI다.

이미지 전체를 보고 판단하되,
반드시 하단 또는 중하단에 있는 "제품 포장 정보" 표만 기준으로 읽어라.

중요:
제품 포장 정보 표의 "EAN" 열에는 실제 EAN 숫자만 있는 것이 아니다.
아래와 같은 값이 들어갈 수 있다.
- 13자리 EAN 숫자
- UPC 숫자
- SKU
- 내부 상품코드
- 문자와 숫자가 섞인 제품코드

따라서 "EAN은 13자리 숫자만"이라고 가정하지 마라.
제품 포장 정보 표의 EAN 열에 보이는 값을 그대로 code로 출력하라.

읽어야 할 항목:
1. wave_no
2. 제품 포장 정보 표의 EAN 열 값 → code
3. 같은 행 오른쪽 끝 "포장수량" 값 → expected_qty

절대 하면 안 되는 것:
- 상단 품목 리스트의 물품 코드를 code로 사용하지 않는다.
- 상단 품목 리스트의 PCS 값을 expected_qty로 사용하지 않는다.
- 하단 합계 수량을 개별 expected_qty로 사용하지 않는다.
- 손글씨, 동그라미, 체크표시 안의 숫자를 수량으로 사용하지 않는다.
- 상품명으로 code를 추정하지 않는다.
- 표 밖의 숫자나 문자는 사용하지 않는다.
- 880으로 시작한다고 가정하지 않는다.
- code가 숫자가 아니라고 해서 제외하지 않는다.

중요 규칙:
- 제품 포장 정보 표의 데이터 행 개수를 먼저 센다.
- 표의 모든 데이터 행을 빠짐없이 추출한다.
- code는 EAN 열에 보이는 값을 그대로 읽는다.
- code가 줄바꿈되어 있으면 한 줄로 합쳐서 읽는다.
- expected_qty는 반드시 같은 행의 "포장수량" 열에서만 가져온다.
- 포장수량이 1이면 반드시 1로 기록한다.
- 코드나 수량이 불명확한 행은 pick_list에 넣지 말고 uncertain_pick_list에 넣는다.
- row_count는 제품 포장 정보 표에서 보이는 데이터 행 개수다.
- pick_list_count는 pick_list에 넣은 행 개수다.
- 반드시 JSON만 출력한다.

JSON 형식:
{
  "wave_no":"0000000000",
  "row_count":2,
  "pick_list_count":2,
  "pick_list":[
    {"code":"8800344689570", "expected_qty":1},
    {"code":"U1BTS26APTOURPOCAR", "expected_qty":1}
  ],
  "uncertain_pick_list":[
    {"raw_text":"불명확한 행 내용", "reason":"code 또는 포장수량이 흐림"}
  ]
}
"""

    return gpt_json(prompt, [image])


def scan_barcodes_from_image(image: Image.Image):
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp가 설치되지 않았습니다.")

    results = zxingcpp.read_barcodes(image)
    codes = []

    for r in results:
        code = normalize_code(str(r.text).strip())
        if len(code) >= 6:
            codes.append(code)

    return codes


def compare_result(pick_data, label_codes, scan_detail):
    expected = {}

    for x in pick_data.get("pick_list", []):
        code = normalize_code(x.get("code", ""))
        if not code:
            continue
        qty = int(x.get("expected_qty", 0))
        expected[code] = expected.get(code, 0) + qty

    actual = Counter([normalize_code(x) for x in label_codes if normalize_code(x)])

    rows = []
    has_error = False

    for code, exp_qty in expected.items():
        act_qty = actual.get(code, 0)
        result = "일치" if exp_qty == act_qty else "불일치"
        if result != "일치":
            has_error = True

        rows.append({
            "code": code,
            "expected_qty": exp_qty,
            "actual_qty": act_qty,
            "result": result
        })

    for code, qty in actual.items():
        if code not in expected:
            has_error = True
            rows.append({
                "code": code,
                "expected_qty": 0,
                "actual_qty": qty,
                "result": "예정에 없음"
            })

    row_count = int(pick_data.get("row_count", 0))
    pick_list_count = len(pick_data.get("pick_list", []))
    uncertain = pick_data.get("uncertain_pick_list", [])

    if row_count != pick_list_count:
        has_error = True

    if uncertain:
        has_error = True

    no_barcode_files = [
        x["file"]
        for x in scan_detail
        if x["status"] == "바코드 미검출"
    ]

    if no_barcode_files:
        has_error = True

    return {
        "status": "확인필요" if has_error else "이상없음",
        "wave_no": pick_data.get("wave_no", ""),
        "rows": rows,
        "row_count": row_count,
        "pick_list_count": pick_list_count,
        "uncertain_pick_list": uncertain,
        "no_barcode_files": no_barcode_files,
        "scan_detail": scan_detail,
        "actual_summary": dict(actual)
    }


@app.get("/")
def home():
    return {
        "message": "Logi Check API is running",
        "endpoint": "/check",
        "method": "POST"
    }


@app.post("/check")
async def check(files: list[UploadFile] = File(...)):
    try:
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "OPENAI_API_KEY가 없습니다."}
            )

        if len(files) < 2:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "피킹리스트 1장 + 라벨사진 1장 이상 필요합니다."}
            )

        images_info = []

        for idx, file in enumerate(files):
            file_bytes = await file.read()
            image = open_image_fixed(file_bytes)
            images_info.append({
                "index": idx,
                "filename": file.filename,
                "image": image
            })

        class_data = classify_images(images_info)

        pick_indexes = [
            x["index"]
            for x in class_data.get("images", [])
            if x.get("type") == "picklist"
        ]

        if not pick_indexes:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "피킹리스트 사진을 찾지 못했습니다.",
                    "classification": class_data
                }
            )

        pick_index = pick_indexes[0]

        label_infos = [
            x for x in images_info
            if x["index"] != pick_index
        ]

        pick_image = images_info[pick_index]["image"]
        pick_data = read_picklist_with_gpt(pick_image)

        all_label_codes = []
        scan_detail = []

        for info in label_infos:
            try:
                codes = scan_barcodes_from_image(info["image"])
                scan_detail.append({
                    "file": info["filename"],
                    "status": "스캔완료" if codes else "바코드 미검출",
                    "codes": codes
                })
                all_label_codes.extend(codes)
            except Exception as e:
                scan_detail.append({
                    "file": info["filename"],
                    "status": "스캔오류",
                    "codes": [],
                    "error": str(e)
                })

        result = compare_result(pick_data, all_label_codes, scan_detail)

        return {
            "ok": True,
            "classification": class_data,
            "picklist_file": images_info[pick_index]["filename"],
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
