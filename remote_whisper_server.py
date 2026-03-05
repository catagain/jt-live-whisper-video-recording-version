#!/usr/bin/env python3
"""
jt-live-whisper 遠端 Whisper ASR 伺服器
部署到 GPU 伺服器，提供 REST API 讓本機上傳音訊檔進行語音辨識。

後端引擎自動偵測：
  1. faster-whisper (CTranslate2 CUDA) — x86_64 GPU，速度最快
  2. openai-whisper (PyTorch CUDA) — aarch64 GPU（如 DGX Spark），也能 GPU 加速
  3. faster-whisper (CPU) — 無 GPU 降級

依賴：faster-whisper, fastapi, uvicorn, python-multipart
      （aarch64 無 CTranslate2 CUDA 時額外需要 openai-whisper）
啟動：python3 server.py [--port 8978] [--host 0.0.0.0]

Author: Jason Cheng (Jason Tools)
"""

import argparse
import os
import tempfile
import time

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="jt-whisper-server")

# ── 偵測最佳後端引擎 ──
_models: dict = {}
_backend = "faster-whisper"  # "faster-whisper" 或 "openai-whisper"
_device = "cpu"
_compute_type = "int8"
_torch_device = "cpu"

if torch.cuda.is_available():
    _torch_device = "cuda"
    # 嘗試 CTranslate2 CUDA（faster-whisper 用）
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            _device = "cuda"
            _compute_type = "float16"
            _backend = "faster-whisper"
            print("[引擎] faster-whisper (CTranslate2 CUDA)")
        else:
            raise RuntimeError("CTranslate2 無 CUDA")
    except Exception:
        # CTranslate2 沒 CUDA，嘗試 openai-whisper（PyTorch CUDA）
        try:
            import whisper as openai_whisper  # noqa: F401
            _backend = "openai-whisper"
            _device = "cuda"
            print("[引擎] openai-whisper (PyTorch CUDA)")
        except ImportError:
            print("[警告] CTranslate2 無 CUDA 且 openai-whisper 未安裝，改用 CPU")
            _backend = "faster-whisper"
else:
    print("[引擎] faster-whisper (CPU)")


# ── 模型載入 ──

def _get_model_faster(model_size: str):
    """faster-whisper 模型"""
    from faster_whisper import WhisperModel
    key = f"fw:{model_size}"
    if key not in _models:
        print(f"[載入模型] {model_size} (faster-whisper, device={_device}, compute={_compute_type})")
        _models[key] = WhisperModel(model_size, device=_device, compute_type=_compute_type)
        print(f"[模型就緒] {model_size}")
    return _models[key]


def _get_model_openai(model_size: str):
    """openai-whisper 模型"""
    import whisper as openai_whisper
    # openai-whisper 模型名稱對應：large-v3-turbo → turbo, large-v3 → large
    name_map = {
        "large-v3-turbo": "turbo",
        "large-v3": "large-v3",
        "medium.en": "medium.en",
        "small.en": "small.en",
        "base.en": "base.en",
    }
    ow_name = name_map.get(model_size, model_size)
    key = f"ow:{ow_name}"
    if key not in _models:
        print(f"[載入模型] {ow_name} (openai-whisper, device={_torch_device})")
        _models[key] = openai_whisper.load_model(ow_name, device=_torch_device)
        print(f"[模型就緒] {ow_name}")
    return _models[key], ow_name


# ── 辨識函式 ──

def _transcribe_faster(wav_path, model_size, language):
    """faster-whisper 辨識"""
    m = _get_model_faster(model_size)
    t0 = time.monotonic()
    segments_iter, info = m.transcribe(wav_path, language=language, beam_size=5, vad_filter=True)
    segments = []
    full_text = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
            full_text.append(text)
    return segments, full_text, round(info.duration, 1), round(time.monotonic() - t0, 1)


def _transcribe_openai(wav_path, model_size, language):
    """openai-whisper 辨識"""
    m, ow_name = _get_model_openai(model_size)
    t0 = time.monotonic()
    result = m.transcribe(wav_path, language=language, beam_size=5)
    segments = []
    full_text = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            segments.append({"start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": text})
            full_text.append(text)
    # openai-whisper 不直接回傳 duration，從最後一段取
    duration = round(segments[-1]["end"], 1) if segments else 0
    return segments, full_text, duration, round(time.monotonic() - t0, 1)


# ── API ──

@app.get("/health")
def health():
    """健康檢查"""
    return {
        "status": "ok",
        "gpu": _device == "cuda",
        "device": _device,
        "backend": _backend,
    }


@app.get("/models")
def list_models():
    """列出已快取的模型"""
    cached = set()
    cached.update(k.split(":", 1)[1] for k in _models.keys())
    # 掃描 HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            name = repo.repo_id
            if name.startswith("Systran/faster-whisper-"):
                cached.add(name.replace("Systran/faster-whisper-", ""))
            elif name.startswith("guillaumekln/faster-whisper-"):
                cached.add(name.replace("guillaumekln/faster-whisper-", ""))
    except Exception:
        pass
    # openai-whisper 模型放在 ~/.cache/whisper/
    whisper_cache = os.path.expanduser("~/.cache/whisper")
    if os.path.isdir(whisper_cache):
        # 檔名格式: large-v3-turbo.pt, medium.en.pt 等
        for f in os.listdir(whisper_cache):
            if f.endswith(".pt"):
                cached.add(f[:-3])
    return {"models": sorted(cached)}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("large-v3-turbo"),
    language: str = Form("en"),
):
    """接收音訊檔，回傳辨識結果"""
    from fastapi.responses import JSONResponse

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        try:
            if _backend == "openai-whisper":
                segments, full_text, duration, proc_time = _transcribe_openai(tmp.name, model, language)
            else:
                segments, full_text, duration, proc_time = _transcribe_faster(tmp.name, model, language)
        except Exception as e:
            print(f"[錯誤] 辨識失敗: {model} — {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"辨識失敗: {model}", "detail": str(e)},
            )

        return {
            "text": " ".join(full_text),
            "segments": segments,
            "language": language,
            "model": model,
            "duration": duration,
            "processing_time": proc_time,
            "device": _device,
            "backend": _backend,
        }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jt-whisper-server")
    parser.add_argument("--port", type=int, default=8978)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"[jt-whisper-server] 啟動 {args.host}:{args.port} (backend={_backend}, device={_device})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
