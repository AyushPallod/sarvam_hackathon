"""
sarvam_demo/sarvam_voice.py
----------------------------
Unified Sarvam AI Voice Service for Disha v2:
  - Speech-To-Text (STT): Sarvam Saaras v3 (`saaras:v3`) with both translit (Hinglish) and transcribe modes.
  - Text-To-Speech (TTS): Sarvam Bulbul v3 (`bulbul:v3`) with natural conversational speaker (default: `ritu`).
  - Chat Completion: Sarvam 105B Conversations (`sarvam-105b-conversations`).
"""

import os
import io
import time
import base64
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Ensure environment variables are loaded
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)
load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

try:
    from sarvamai import SarvamAI
    _client = SarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY else None
except Exception:
    _client = None


def get_client() -> Optional[Any]:
    global _client
    if _client is None:
        key = os.getenv("SARVAM_API_KEY")
        if key:
            from sarvamai import SarvamAI
            _client = SarvamAI(api_subscription_key=key)
    return _client


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "input.webm",
    content_type: str = "audio/webm",
    language_code: str = "hi-IN",
    mode: str = "translit"
) -> Dict[str, Any]:
    """
    Transcribes audio using Sarvam Saaras v3 STT.
    Returns transcript, detected language, and latency.
    """
    client = get_client()
    if not client:
        return {
            "error": "SARVAM_API_KEY not configured",
            "transcript": "",
            "latency_ms": 0.0
        }

    t0 = time.perf_counter()
    bio = io.BytesIO(audio_bytes)
    bio.name = filename

    try:
        response = client.speech_to_text.transcribe(
            file=(filename, bio, content_type),
            model="saaras:v3",
            language_code=language_code,
            mode=mode
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "transcript": response.transcript or "",
            "language_code": response.language_code or language_code,
            "latency_ms": round(latency_ms, 1),
            "request_id": response.request_id
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "error": str(e),
            "transcript": "",
            "latency_ms": round(latency_ms, 1)
        }


def synthesize_speech(
    text: str,
    speaker: str = "ritu",
    language_code: str = "hi-IN",
    model: str = "bulbul:v3"
) -> Dict[str, Any]:
    """
    Synthesizes speech audio using Sarvam Bulbul v3 TTS.
    Returns base64 audio and latency.
    """
    client = get_client()
    if not client:
        return {
            "error": "SARVAM_API_KEY not configured",
            "audio_base64": None,
            "latency_ms": 0.0
        }

    t0 = time.perf_counter()
    # Strip any markup or markdown that shouldn't be read aloud
    clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    if not clean_text:
        return {"audio_base64": None, "latency_ms": 0.0}

    # Bulbul v3 has high quality conversational speakers: ritu, priya, roopa, simran, etc.
    try:
        response = client.text_to_speech.convert(
            text=clean_text[:500],  # keep audio chunk concise
            language_code=language_code,
            speaker=speaker,
            model=model
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        audios = response.audios or []
        audio_b64 = audios[0] if audios else None

        return {
            "audio_base64": audio_b64,
            "speaker": speaker,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "request_id": response.request_id
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "error": str(e),
            "audio_base64": None,
            "latency_ms": round(latency_ms, 1)
        }
