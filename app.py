import streamlit as st
import os
import base64
import json
import traceback
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

# SSL 인증서 문제 해결
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from openai import OpenAI

# 바코드 스캔
try:
    import zxingcpp
except Exception:
    zxingcpp = None


# =========================
# 기본 설정
# =========================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="물류 사진 자동 검수봇",
    layout="wide"
)

st.title("📦 물류 사진 자동 검수봇")
st.caption("피킹리스트는 GPT가 읽고, 상품라벨은 바코드 스캐너가 실제 바코드를 읽습니다.")

if not api_key:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
# 이미지 Base64 변환
# =========================

def img64(file):
    return base64.b64encode(file.getvalue()).decode("utf-8")


# =========================
# 피킹리스트 GPT 분석
# =========================

def read_picklist_with_gpt(pick_img):
    prompt = """
너는 물류 피킹리스트 판독 AI다.

피킹리스트 이미지에서 아래 항목만 추출한다.

1. wave_no
2. 제품 포장 정보 영역의 EAN
3. 각 EAN별 포장수량

중요 규칙:
- 피킹리스트의 제품 포장 정보 표를 우선한다.
- 상품명으로 EAN을 추정하지 않는다.
- 숫자가 불명확하면 추측하지 말고 제외한다.
- 반드시 JSON만 출력한다.
- 설명 문장은 출력하지 않는다.

JSON 형식:
{
  "wave_no":"0000000000",
  "pick_list":[
    {
      "ean":"8800000000000",
      "expected_qty":2
    }
  ]
}
"""

    content = [
        {
            "type": "input_text",
            "text": prompt
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{img64(pick_img)}"
        }
    ]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    result = response.output_text.strip()
    result = result.replace("```json", "").replace("```", "").strip()

    return json.loads(result)


# =========================
# 상품라벨 바코드 스캔
# =========================

def scan_barcodes_from_image(uploaded_file):
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp가 설치되지 않았습니다. requirements.txt에 zxing-cpp를 추가해주세요.")

    image = Image.open(uploaded_file).convert("RGB")

    results = zxingcpp.read_barcodes(image)

    barcodes = []

    for r in results:
        value = str(r.text).strip()

        # EAN은 보통 13자리. 혹시 앞뒤 공백/문자 제거
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
            eans = []
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


# =========================
# 업로드 화면
# =========================

pick_img = st.file_uploader(
    "피킹리스트 사진",
    type=["png", "jpg", "jpeg"]
)

label_imgs = st.file_uploader(
    "상품라벨 사진 (여러 장 가능)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# =========================
# 검수
# =========================

if st.button("검수 시작"):

    if not pick_img:
        st.warning("피킹리스트 사진을 업로드해주세요.")
        st.stop()

    if len(label_imgs) == 0:
        st.warning("상품라벨 사진을 업로드해주세요.")
        st.stop()

    # 1. 피킹리스트 GPT 판독
    try:
        with st.spinner("피킹리스트 분석 중..."):
            pick_data = read_picklist_with_gpt(pick_img)

    except Exception as e:
        st.error("피킹리스트 AI 분석 실패")
        st.write(str(e))
        st.code(traceback.format_exc())
        st.stop()

    # 2. 상품라벨 바코드 스캔
    try:
        with st.spinner("상품라벨 바코드 스캔 중..."):
            label_eans, scan_detail = scan_all_label_images(label_imgs)

    except Exception as e:
        st.error("상품라벨 바코드 스캔 실패")
        st.write(str(e))
        st.code(traceback.format_exc())
        st.stop()

    # 3. 비교
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

        rows.append(
            {
                "EAN": ean,
                "예정수량": exp_qty,
                "사진수량": act_qty,
                "결과": result
            }
        )

    for ean, qty in actual.items():

        if ean not in expected:

            has_error = True

            rows.append(
                {
                    "EAN": ean,
                    "예정수량": 0,
                    "사진수량": qty,
                    "결과": "예정에 없음"
                }
            )

    # 바코드 미검출 파일이 있으면 확인필요 처리
    no_barcode_files = [
        x["file"]
        for x in scan_detail
        if x["status"] == "바코드 미검출"
    ]

    if no_barcode_files:
        has_error = True

    # 4. 결과 출력
    st.subheader("검수 결과")

    st.write(f"웨이브 : {pick_data.get('wave_no', '')}")

    if has_error:
        st.error("확인필요")
    else:
        st.success("이상없음")

    result_df = pd.DataFrame(rows)

    st.dataframe(
        result_df,
        use_container_width=True
    )

    if no_barcode_files:
        st.warning("바코드 미검출 파일이 있습니다. 해당 사진은 수동 확인이 필요합니다.")
        for f in no_barcode_files:
            st.write(f"- {f}")

    with st.expander("상품라벨 바코드 스캔 상세"):
        st.json(scan_detail)

    with st.expander("피킹리스트 GPT 추출 원본"):
        st.json(pick_data)

    with st.expander("상품라벨 EAN 집계"):
        st.json(dict(actual))
