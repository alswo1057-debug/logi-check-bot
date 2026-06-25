import streamlit as st
import os
import base64
import json
import traceback
from collections import Counter
from io import BytesIO

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageOps

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from openai import OpenAI

try:
    import zxingcpp
except Exception:
    zxingcpp = None


MODEL_NAME = "gpt-5.4"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="물류 사진 자동 검수봇", layout="wide")
st.title("📦 물류 사진 자동 검수봇")
st.caption("피킹리스트 전체 이미지를 GPT가 읽고, 상품라벨은 실제 바코드를 스캔합니다.")

if not api_key:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

client = OpenAI(api_key=api_key)


def open_image_fixed(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def image_to_base64(uploaded_file):
    image = open_image_fixed(uploaded_file)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def read_picklist_with_gpt(pick_img):
    prompt = """
너는 물류 피킹리스트 판독 AI다.

이미지 전체를 보고 판단하되,
반드시 하단 또는 중하단에 있는 "제품 포장 정보" 표만 기준으로 읽어라.

읽어야 할 항목:
1. wave_no
2. 제품 포장 정보 표의 EAN
3. 제품 포장 정보 표의 오른쪽 끝 "포장수량" 값

절대 하면 안 되는 것:
- 상단 품목 리스트의 PCS 값을 expected_qty로 사용하지 않는다.
- 상단 품목 리스트의 PCS 21, 42, 84 같은 숫자를 사용하지 않는다.
- 하단 합계 수량을 개별 EAN expected_qty로 사용하지 않는다.
- 손글씨, 동그라미, 체크표시 안의 숫자를 수량으로 사용하지 않는다.
- 상품명으로 EAN을 추정하지 않는다.
- 표 밖의 숫자는 사용하지 않는다.

중요 규칙:
- 제품 포장 정보 표의 데이터 행 개수를 먼저 센다.
- 표의 모든 데이터 행을 빠짐없이 추출한다.
- EAN은 13자리 숫자만 인정한다.
- expected_qty는 반드시 같은 행의 "포장수량" 열에서만 가져온다.
- 포장수량이 동그라미 쳐져 있어도 인쇄된 숫자를 읽는다.
- 포장수량이 1이면 반드시 1로 기록한다.
- 숫자가 불명확한 행은 pick_list에 넣지 말고 uncertain_pick_list에 넣는다.
- row_count는 제품 포장 정보 표에서 보이는 데이터 행 개수다.
- pick_list_count는 pick_list에 넣은 행 개수다.
- 반드시 JSON만 출력한다.
- 설명 문장은 출력하지 않는다.

JSON 형식:
{
  "wave_no":"0000000000",
  "row_count":3,
  "pick_list_count":3,
  "pick_list":[
    {
      "ean":"8800000000000",
      "expected_qty":1
    }
  ],
  "uncertain_pick_list":[
    {
      "raw_text":"불명확한 행 내용",
      "reason":"EAN 또는 포장수량이 흐림"
    }
  ]
}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_to_base64(pick_img)}"
                    }
                ]
            }
        ]
    )

    text = clean_json_text(response.output_text)
    return json.loads(text)


def scan_barcodes_from_image(uploaded_file):
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp가 설치되지 않았습니다.")

    image = open_image_fixed(uploaded_file)
    results = zxingcpp.read_barcodes(image)

    barcodes = []

    for r in results:
        value = str(r.text).strip()
        digits = "".join([c for c in value if c.isdigit()])

        if len(digits) in [12, 13, 14]:
            barcodes.append(digits)

    return barcodes


def scan_all_label_images(label_imgs):
    all_eans = []
    scan_detail = []

    for img in label_imgs:
        try:
            eans = scan_barcodes_from_image(img)
        except Exception as e:
            scan_detail.append({
                "file": img.name,
                "status": "스캔오류",
                "eans": [],
                "error": str(e)
            })
            continue

        scan_detail.append({
            "file": img.name,
            "status": "스캔완료" if eans else "바코드 미검출",
            "eans": eans
        })

        all_eans.extend(eans)

    return all_eans, scan_detail


pick_img = st.file_uploader(
    "피킹리스트 사진",
    type=["png", "jpg", "jpeg"]
)

label_imgs = st.file_uploader(
    "상품라벨 사진 (여러 장 가능)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


if st.button("검수 시작"):

    if not pick_img:
        st.warning("피킹리스트 사진을 업로드해주세요.")
        st.stop()

    if len(label_imgs) == 0:
        st.warning("상품라벨 사진을 업로드해주세요.")
        st.stop()

    try:
        with st.spinner("피킹리스트 분석 중..."):
            pick_data = read_picklist_with_gpt(pick_img)

    except Exception as e:
        st.error("피킹리스트 AI 분석 실패")
        st.write(str(e))
        st.code(traceback.format_exc())
        st.stop()

    try:
        with st.spinner("상품라벨 바코드 스캔 중..."):
            label_eans, scan_detail = scan_all_label_images(label_imgs)

    except Exception as e:
        st.error("상품라벨 바코드 스캔 실패")
        st.write(str(e))
        st.code(traceback.format_exc())
        st.stop()

    expected = {
        str(x["ean"]): int(x["expected_qty"])
        for x in pick_data.get("pick_list", [])
        if x.get("ean")
    }

    actual = Counter([str(x) for x in label_eans])

    rows = []
    has_error = False

    for ean, exp_qty in expected.items():
        act_qty = actual.get(ean, 0)
        result = "일치" if exp_qty == act_qty else "불일치"

        if result == "불일치":
            has_error = True

        rows.append({
            "EAN": ean,
            "예정수량": exp_qty,
            "사진수량": act_qty,
            "결과": result
        })

    for ean, qty in actual.items():
        if ean not in expected:
            has_error = True
            rows.append({
                "EAN": ean,
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

    st.subheader("검수 결과")
    st.write(f"웨이브 : {pick_data.get('wave_no', '')}")

    if has_error:
        st.error("확인필요")
    else:
        st.success("이상없음")

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)

    if row_count != pick_list_count:
        st.warning(
            f"피킹리스트 행 개수 확인필요: 제품 포장 정보 표 행 {row_count}개 / 추출 {pick_list_count}개"
        )

    if uncertain_pick_list:
        st.warning("피킹리스트에서 불명확한 행이 있습니다.")
        st.json(uncertain_pick_list)

    if no_barcode_files:
        st.warning("바코드 미검출 파일이 있습니다. 해당 사진은 수동 확인이 필요합니다.")
        for f in no_barcode_files:
            st.write(f"- {f}")

    with st.expander("피킹리스트 GPT 추출 원본"):
        st.json(pick_data)

    with st.expander("상품라벨 바코드 스캔 상세"):
        st.json(scan_detail)

    with st.expander("상품라벨 EAN 집계"):
        st.json(dict(actual))
