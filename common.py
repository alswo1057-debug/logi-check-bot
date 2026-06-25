import os
import base64
import json
import traceback
from collections import Counter
from io import BytesIO

from PIL import Image, ImageOps
from openai import OpenAI

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

try:
    import zxingcpp
except Exception:
    zxingcpp = None


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


def open_image_fixed(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def open_image_fixed_from_bytes(file_bytes):
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_base64(uploaded_file):
    image = open_image_fixed(uploaded_file)
    return pil_to_base64(image)


def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def normalize_code(value):
    """
    비교용 코드 정규화.
    - 공백, 줄바꿈, 하이픈 제거
    - 대문자 변환
    - 숫자 EAN/UPC, 문자 SKU 모두 허용
    """
    if value is None:
        return ""

    value = str(value).strip().upper()
    remove_chars = [" ", "\n", "\r", "\t", "-", "_"]
    for ch in remove_chars:
        value = value.replace(ch, "")

    return value


def call_gpt_with_images(prompt, images):
    if not client:
        raise RuntimeError("OPENAI_API_KEY를 찾을 수 없습니다.")

    content = [{"type": "input_text", "text": prompt}]

    for image in images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{pil_to_base64(image)}"
        })

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    text = clean_json_text(response.output_text)
    return json.loads(text)


def read_picklist_with_gpt(pick_img):
    """
    기존 Streamlit 최종 코드와 같은 프롬프트.
    pick_img는 Streamlit UploadedFile 또는 PIL Image 둘 다 가능.
    """
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

작업 순서:
1. 먼저 이미지에서 "제품 포장 정보" 표 위치를 찾는다.
2. 표의 헤더 행을 확인한다.
3. 표의 데이터 행을 위에서 아래 순서대로 하나씩 검토한다.
4. 각 행마다 EAN 열의 code와 같은 행의 포장수량만 읽는다.
5. 모든 행을 확인한 뒤 row_count와 pick_list_count가 일치하는지 검증한다.
6. 누락된 행이 없는지 마지막으로 한 번 더 검토한다.
7. 최종 결과만 JSON으로 출력한다.

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
- 설명 문장은 출력하지 않는다.

JSON 형식:
{
  "wave_no":"0000000000",
  "row_count":2,
  "pick_list_count":2,
  "pick_list":[
    {
      "code":"8800344689570",
      "expected_qty":1
    },
    {
      "code":"U1BTS26APTOURPOCAR",
      "expected_qty":1
    }
  ],
  "uncertain_pick_list":[
    {
      "raw_text":"불명확한 행 내용",
      "reason":"code 또는 포장수량이 흐림"
    }
  ]
}
"""

    if isinstance(pick_img, Image.Image):
        image = pick_img
    else:
        image = open_image_fixed(pick_img)

    return call_gpt_with_images(prompt, [image])


def classify_images_with_gpt(images_info):
    """
    FastAPI용.
    여러 이미지 중 피킹리스트 1장을 찾고 나머지는 라벨로 분류.
    """
    prompt = """
너는 물류 검수 사진 분류 AI다.

여러 장의 이미지가 순서대로 제공된다.
각 이미지가 아래 중 무엇인지 분류하라.

분류값:
- picklist: 피킹리스트, 출고지시서, Wave No, 제품 포장 정보 표, EAN, 포장수량이 보이는 문서 사진
- label: 상품 라벨, 바코드 라벨, 제품 실물 바코드 사진
- unknown: 판단 불가

중요 규칙:
- 피킹리스트는 보통 A4 문서처럼 보이고 표가 많다.
- 피킹리스트에는 Wave No, 제품 포장 정보, EAN, 포장수량 같은 문구가 보일 수 있다.
- 상품라벨은 보통 바코드, QR, CODE, SKU, 상품명 라벨이 크게 보인다.
- 피킹리스트가 여러 장으로 판단되면 가장 피킹리스트 가능성이 높은 1장만 picklist로 둔다.
- 반드시 JSON만 출력한다.
- 설명 문장은 출력하지 않는다.

JSON 형식:
{
  "images":[
    {"index":0, "type":"picklist", "reason":"제품 포장 정보 표가 보임"},
    {"index":1, "type":"label", "reason":"바코드 라벨 사진"}
  ]
}
"""

    images = [x["image"] for x in images_info]
    return call_gpt_with_images(prompt, images)


def scan_barcodes_from_image(uploaded_file_or_image):
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp가 설치되지 않았습니다.")

    if isinstance(uploaded_file_or_image, Image.Image):
        image = uploaded_file_or_image
    else:
        image = open_image_fixed(uploaded_file_or_image)

    results = zxingcpp.read_barcodes(image)

    codes = []

    for r in results:
        value = str(r.text).strip()
        code = normalize_code(value)

        if len(code) >= 6:
            codes.append(code)

    return codes


def scan_all_label_images(label_imgs):
    all_codes = []
    scan_detail = []

    for img in label_imgs:
        filename = getattr(img, "name", None) or getattr(img, "filename", None) or "unknown"

        try:
            codes = scan_barcodes_from_image(img)
        except Exception as e:
            scan_detail.append({
                "file": filename,
                "status": "스캔오류",
                "codes": [],
                "error": str(e)
            })
            continue

        scan_detail.append({
            "file": filename,
            "status": "스캔완료" if codes else "바코드 미검출",
            "codes": codes
        })

        all_codes.extend(codes)

    return all_codes, scan_detail


def build_compare_result(pick_data, label_codes, scan_detail):
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

        if result == "불일치":
            has_error = True

        rows.append({
            "CODE": code,
            "예정수량": exp_qty,
            "사진수량": act_qty,
            "결과": result
        })

    for code, qty in actual.items():
        if code not in expected:
            has_error = True
            rows.append({
                "CODE": code,
                "예정수량": 0,
                "사진수량": qty,
                "결과": "예정에 없음"
            })

    row_count = int(pick_data.get("row_count", 0))
    pick_list_count = len(pick_data.get("pick_list", []))
    uncertain_pick_list = pick_data.get("uncertain_pick_list", [])

    if row_count != pick_list_count:
        has_error = True

    if uncertain_pick_list:
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
        "has_error": has_error,
        "wave_no": pick_data.get("wave_no", ""),
        "rows": rows,
        "row_count": row_count,
        "pick_list_count": pick_list_count,
        "uncertain_pick_list": uncertain_pick_list,
        "no_barcode_files": no_barcode_files,
        "scan_detail": scan_detail,
        "actual_summary": dict(actual),
        "pick_data": pick_data
    }


def check_by_files(pick_img, label_imgs):
    pick_data = read_picklist_with_gpt(pick_img)
    label_codes, scan_detail = scan_all_label_images(label_imgs)
    return build_compare_result(pick_data, label_codes, scan_detail)


def check_by_auto_classification(uploaded_files):
    """
    FastAPI에서 여러 파일을 받았을 때:
    1. 이미지 로딩
    2. GPT로 피킹리스트 자동 분류
    3. 나머지 라벨 스캔
    4. 비교
    """
    images_info = []

    for idx, file_obj in enumerate(uploaded_files):
        images_info.append({
            "index": idx,
            "filename": file_obj["filename"],
            "image": file_obj["image"]
        })

    class_data = classify_images_with_gpt(images_info)

    pick_indexes = [
        x["index"]
        for x in class_data.get("images", [])
        if x.get("type") == "picklist"
    ]

    if not pick_indexes:
        raise RuntimeError("피킹리스트 사진을 찾지 못했습니다.")

    pick_index = pick_indexes[0]

    pick_image = images_info[pick_index]["image"]

    label_images = []
    for info in images_info:
        if info["index"] != pick_index:
            img = info["image"]
            img.filename = info["filename"]
            label_images.append(img)

    result = check_by_files(pick_image, label_images)

    result["classification"] = class_data
    result["picklist_file"] = images_info[pick_index]["filename"]

    return result
