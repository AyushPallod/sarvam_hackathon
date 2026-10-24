"""
sarvam_demo/run_sarvam_voice_demo.py
------------------------------------
CLI Voice Demonstration using Sarvam AI:
  1. User speaks or provides an utterance in Hindi / Hinglish.
  2. Disha v2 Parallel Pipeline processes the turn (Safety, Router, Retrieval, DB, LLM, Guardrails).
  3. Sarvam Bulbul v3 TTS synthesizes spoken audio using Ritu's voice.
  4. Audio is saved locally to output_disha_reply.wav and played/verified.
"""

import os
import sys
import time
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from disha_v2.orchestrator import DishaOrchestrator, warmup
import sarvam_voice

BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RED    = "\033[91m"
RESET  = "\033[0m"

def run_voice_turn(patient_id: str, utterance: str, speaker: str = "ritu"):
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{BLUE}🎙️ Patient [{patient_id}]:{RESET} \"{utterance}\"")
    print(f"{CYAN}{'='*65}{RESET}")

    t0 = time.perf_counter()
    orch = DishaOrchestrator(patient_id)

    # 1. Pipeline execution
    result = orch.process_turn(utterance)
    pipe_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n{BOLD}Pipeline Breakdown:{RESET}")
    print(f"  Safety (MiniLM+AC): {result.safety.get('total_latency_ms', 0):.1f} ms (fired={result.safety.get('fired')})")
    print(f"  Router (Laya v2):   {result.routing.get('laya_latency_ms', 0):.1f} ms (domain={result.routing.get('domain')}, tool={result.routing.get('tool_type')})")
    if result.tool_result:
        print(f"  Tool Executed:      {result.tool_result.get('_tool_name')} ({result.tool_result.get('_latency_ms', 0):.1f} ms)")
    print(f"  LLM TTFT:           {result.timing.get('llm_ttft_ms', 0):.1f} ms")
    print(f"  Guardrail:          {result.timing.get('guardrail_ms', 0):.2f} ms")
    print(f"  Pipeline Total:     {result.total_ms:.1f} ms")

    print(f"\n{BOLD}Disha Response:{RESET}\n  \"{result.final_response}\"")

    # 2. Sarvam TTS
    print(f"\n{BOLD}{YELLOW}Synthesizing voice with Sarvam Bulbul v3 (speaker: {speaker})...{RESET}")
    tts_res = sarvam_voice.synthesize_speech(result.final_response, speaker=speaker)

    tts_ms = tts_res.get("latency_ms", 0.0)
    audio_b64 = tts_res.get("audio_base64")

    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        out_wav = os.path.join(BASE_DIR, "output_disha_reply.wav")
        with open(out_wav, "wb") as f:
            f.write(audio_bytes)
        print(f"  {GREEN}✅ TTS Synthesis Success!{RESET}")
        print(f"  TTS Latency: {tts_ms:.1f} ms")
        print(f"  Saved Spoken WAV: {out_wav} ({len(audio_bytes):,} bytes)")
        print(f"  {BOLD}{GREEN}TOTAL END-TO-END SPOKEN LATENCY: {result.total_ms + tts_ms:.1f} ms{RESET}")
    else:
        print(f"  {RED}TTS Error: {tts_res.get('error')}{RESET}")


def main():
    print(f"\n{BOLD}{CYAN}Warm-up ML Pipeline & Sarvam Voice Client...{RESET}")
    warmup()

    # Test Turn 1: Pricing query
    run_voice_turn(
        patient_id="P-REEMA-01",
        utterance="Yeh app paid hai kya? Kitna lagta hai?",
        speaker="ritu"
    )

    # Test Turn 2: Emergency detection
    run_voice_turn(
        patient_id="P-REEMA-01",
        utterance="Aaj workout ke dauran seene mein dard tha aur left haath sunn ho gaya",
        speaker="ritu"
    )

if __name__ == "__main__":
    main()
