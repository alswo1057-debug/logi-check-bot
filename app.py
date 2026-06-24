import streamlit as st
import os
import base64
import json
import traceback
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

# SSL 인증서 문제 해결
try:
    import truststore
    truststore.inject_into_ssl()
except:
    pass

from openai import OpenAI

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

if not api_key:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# 이미지 Base64 변환
# =========================

def img64(file):
    return base64.b64encode(
        file.getvalue()
    ).decode("utf-8")

# =========================
# GPT 분석
# =========================

def read_images(pick_img, label_imgs):

    prompt = """
너는 물류 검수 AI다.

1. 피킹리스트에서 EAN과 예정수량을 추출한다.
2. 상품라벨 사진은 여러 장일 수 있다.
3. 모든 상품라벨 사진에서 EAN을 추출한다.
4. 동일 EAN은 모두 기록한다.
5. 반드시 JSON만 출력한다.

형식:

{
  "wave_no":"12345",
  "pick_list":[
    {
      "ean":"8800000000000",
      "expected_qty":2
    }
  ],
  "label_eans":[
    "8800000000000",
    "8800000000000"
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

    for img in label_imgs:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img64(img)}"
            }
        )

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

    result = (
        result.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(result)

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

    try:

        with st.spinner("AI 검수 중..."):
            data = read_images(
                pick_img,
                label_imgs
            )

    except Exception as e:

        st.error("AI 분석 실패")

        st.write(str(e))

        st.code(
            traceback.format_exc()
        )

        st.stop()

    expected = {
        str(x["ean"]): int(x["expected_qty"])
        for x in data["pick_list"]
    }

    actual = Counter(
        [str(x) for x in data["label_eans"]]
    )

    rows = []

    has_error = False

    for ean, exp_qty in expected.items():

        act_qty = actual.get(ean, 0)

        result = (
            "일치"
            if exp_qty == act_qty
            else "불일치"
        )

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

    st.subheader("검수 결과")

    st.write(
        f"웨이브 : {data.get('wave_no','')}"
    )

    if has_error:
        st.error("확인필요")
    else:
        st.success("이상없음")

    result_df = pd.DataFrame(rows)

    st.dataframe(
        result_df,
        use_container_width=True
    )

    with st.expander("GPT 추출 원본"):
        st.json(data)