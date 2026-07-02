"""
카카오톡 채널 챗봇용 스킬 서버.

흐름:
1. 사용자가 채널에 사진(피킹리스트 + 상품라벨)을 보냄
   -> 오픈빌더 "이미지 보안전송" 플러그인(@sys.plugin.secureimage)이 이미지 URL을 스킬로 전달
2. 이 서버가 요청을 받아서 즉시 "검수 중이에요" 응답 (콜백 모드)
3. 백그라운드에서 기존 common.py 로직으로 실제 검수 수행
4. 완료되면 callbackUrl로 결과를 다시 전송
5. (선택) 결과를 구글시트 웹훅으로도 기록

주의:
- extract_image_urls()의 파라미터 키 이름은 실제 오픈빌더에서 설정한
  파라미터명에 맞춰 수정이 필요할 수 있음. 처음엔 로그를 찍어서
  실제 payload 구조를 확인한 뒤 맞추는 걸 권장.
- 콜백 API는 카카오 승인(AI 챗봇 전환)이 필요함. 승인 전에는
  동기 처리로 동작하며, GPT 응답이 느리면 타임아웃 날 수 있음(테스트용).
"""

import os
import json
import logging
import threading

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from common import (
    MODEL_NAME,
    open_image_fixed_from_bytes,
    check_by_auto_classification,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kakao_skill")

app = FastAPI(title="Logi Check Kakao Skill")

GOOGLE_SHEET_WEBHOOK_URL = os.getenv("GOOGLE_SHEET_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# 카카오 요청 파싱
# ---------------------------------------------------------------------------

IMAGE_PARAM_NAME = "사진검수"  # 오픈빌더에서 만든 파라미터명 (플러그인: sys.plugin.secureimage)


def _to_url_list(value):
    """단일 URL 문자열 / URL 리스트 / JSON 문자열 등을 URL 리스트로 정규화."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.startswith("http")]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, str) and v.startswith("http")]
            except Exception:
                pass
        if stripped.startswith("http"):
            return [stripped]
    return []


def extract_image_urls(payload: dict):
    """
    스킬 request에서 이미지 보안전송 플러그인으로 전달된 이미지 URL 목록을 추출.
    파라미터명은 오픈빌더에서 "사진검수"로 만들었으므로 이를 우선 확인하고,
    구조가 다르게 오는 경우를 대비해 detailParams/params 전체도 훑는다.
    """
    action = payload.get("action", {})
    params = action.get("params", {}) or {}
    detail_params = action.get("detailParams", {}) or {}

    logger.info("skill request action.params: %s", params)
    logger.info("skill request action.detailParams: %s", detail_params)

    # 1순위: 정확한 파라미터명으로 바로 조회
    if IMAGE_PARAM_NAME in params:
        urls = _to_url_list(params[IMAGE_PARAM_NAME])
        if urls:
            return urls

    if IMAGE_PARAM_NAME in detail_params:
        dp = detail_params[IMAGE_PARAM_NAME]
        origin = dp.get("origin") if isinstance(dp, dict) else None
        urls = _to_url_list(origin)
        if urls:
            return urls

    # 2순위(fallback): detailParams 전체를 훑어서 origin에 이미지 URL이 있는지 확인
    for value in detail_params.values():
        origin = value.get("origin") if isinstance(value, dict) else None
        if not origin:
            continue
        parsed = origin
        if isinstance(origin, str) and origin.strip().startswith("["):
            try:
                parsed = json.loads(origin)
            except Exception:
                parsed = origin
        if isinstance(parsed, list) and parsed:
            return parsed
        if isinstance(parsed, str) and parsed.startswith("http"):
            return [parsed]

    # params에서 리스트/URL 문자열 직접 확인
    for value in params.values():
        if isinstance(value, list) and value:
            return value
        if isinstance(value, str) and value.startswith("http"):
            return [value]

    return []


def get_user_id(payload: dict) -> str:
    return (
        payload.get("userRequest", {})
        .get("user", {})
        .get("id", "unknown")
    )


def get_callback_url(payload: dict):
    return payload.get("userRequest", {}).get("callbackUrl")


# ---------------------------------------------------------------------------
# 검수 처리
# ---------------------------------------------------------------------------

def download_image(url: str):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return open_image_fixed_from_bytes(resp.content)


def format_result_text(result: dict) -> str:
    header = "⚠️ 확인필요" if result.get("has_error") else "✅ 이상없음"
    lines = [header, f"웨이브: {result.get('wave_no', '-')}", ""]

    for row in result.get("rows", []):
        mark = "✅" if row["결과"] == "일치" else "❌"
        lines.append(f"{mark} {row['CODE']} 예정 {row['예정수량']} / 실제 {row['사진수량']}")

    if result.get("no_barcode_files"):
        lines.append("")
        lines.append("⚠️ 바코드 미검출 사진 있음 (수동확인 필요)")

    if result.get("uncertain_pick_list"):
        lines.append("")
        lines.append("⚠️ 피킹리스트에 불명확한 행 있음")

    return "\n".join(lines)


def log_to_sheet(user_id: str, result: dict = None, error: str = None):
    if not GOOGLE_SHEET_WEBHOOK_URL:
        return
    try:
        requests.post(
            GOOGLE_SHEET_WEBHOOK_URL,
            json={
                "user_id": user_id,
                "wave_no": (result or {}).get("wave_no", ""),
                "status": (result or {}).get("status", "오류" if error else ""),
                "error": error or "",
                "detail": result or {},
            },
            timeout=10,
        )
    except Exception:
        logger.exception("구글시트 로그 실패")


def simple_text_response(text: str) -> dict:
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": text}}]},
    }


def send_callback(callback_url: str, text: str):
    try:
        requests.post(callback_url, json=simple_text_response(text), timeout=15)
    except Exception:
        logger.exception("콜백 전송 실패")


def process_and_callback(image_urls, callback_url, user_id):
    try:
        uploaded_files = []
        for idx, url in enumerate(image_urls):
            image = download_image(url)
            uploaded_files.append({"filename": f"kakao_{idx + 1}.jpg", "image": image})

        result = check_by_auto_classification(uploaded_files)
        text = format_result_text(result)
        log_to_sheet(user_id, result=result)

    except Exception as e:
        logger.exception("검수 처리 실패")
        text = f"❌ 검수 중 오류가 발생했어요: {e}"
        log_to_sheet(user_id, error=str(e))

    send_callback(callback_url, text)


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"ok": True, "message": "Kakao skill server running", "model": MODEL_NAME}


@app.post("/kakao/skill")
async def kakao_skill(request: Request):
    payload = await request.json()
    logger.info("raw payload: %s", json.dumps(payload, ensure_ascii=False)[:2000])

    user_id = get_user_id(payload)
    callback_url = get_callback_url(payload)
    image_urls = extract_image_urls(payload)

    if not image_urls:
        return JSONResponse(simple_text_response(
            "사진을 찾지 못했어요. 피킹리스트 사진과 상품라벨 사진을 함께 보내주세요."
        ))

    if len(image_urls) < 2:
        return JSONResponse(simple_text_response(
            "사진이 최소 2장 필요해요. 피킹리스트 1장 + 상품라벨 1장 이상 보내주세요."
        ))

    if not callback_url:
        # 콜백 미승인 상태 - 동기 처리 (테스트용, GPT 응답이 느리면 타임아웃 가능)
        try:
            uploaded_files = []
            for idx, url in enumerate(image_urls):
                image = download_image(url)
                uploaded_files.append({"filename": f"kakao_{idx + 1}.jpg", "image": image})
            result = check_by_auto_classification(uploaded_files)
            text = format_result_text(result)
            log_to_sheet(user_id, result=result)
        except Exception as e:
            logger.exception("동기 처리 실패")
            text = f"❌ 검수 중 오류가 발생했어요: {e}"
            log_to_sheet(user_id, error=str(e))

        return JSONResponse(simple_text_response(text))

    # 콜백 승인된 경우 - 즉시 응답 후 백그라운드 처리
    threading.Thread(
        target=process_and_callback,
        args=(image_urls, callback_url, user_id),
        daemon=True,
    ).start()

    return JSONResponse({
        "version": "2.0",
        "useCallback": True,
        "data": {"text": "📦 사진 확인했어요! 검수 중입니다. 완료되면 바로 알려드릴게요."},
    })
