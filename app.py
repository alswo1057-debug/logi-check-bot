import streamlit as st
import os
import traceback

import pandas as pd
from dotenv import load_dotenv

from common import (
    MODEL_NAME,
    check_by_files
)


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="물류 사진 자동 검수봇", layout="wide")
st.title("📦 물류 사진 자동 검수봇")
st.caption("피킹리스트의 상품 식별코드와 수량을 GPT가 읽고, 상품라벨은 실제 바코드를 스캔합니다.")

if not api_key:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

st.caption(f"사용 모델: {MODEL_NAME}")

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
        with st.spinner("피킹리스트 분석 및 상품라벨 바코드 스캔 중..."):
            result = check_by_files(pick_img, label_imgs)

    except Exception as e:
        st.error("검수 실패")
        st.write(str(e))
        st.code(traceback.format_exc())
        st.stop()

    st.subheader("검수 결과")
    st.write(f"웨이브 : {result.get('wave_no', '')}")

    if result.get("has_error"):
        st.error("확인필요")
    else:
        st.success("이상없음")

    result_df = pd.DataFrame(result.get("rows", []))
    st.dataframe(result_df, use_container_width=True)

    row_count = result.get("row_count", 0)
    pick_list_count = result.get("pick_list_count", 0)
    uncertain_pick_list = result.get("uncertain_pick_list", [])
    no_barcode_files = result.get("no_barcode_files", [])

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
        st.json(result.get("pick_data", {}))

    with st.expander("상품라벨 바코드 스캔 상세"):
        st.json(result.get("scan_detail", []))

    with st.expander("상품라벨 CODE 집계"):
        st.json(result.get("actual_summary", {}))
