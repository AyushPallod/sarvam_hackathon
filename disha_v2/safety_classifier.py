"""
disha_v2/safety_classifier.py
-------------------------------
Class D — Safety-critical detection.

Architecture per solution_v2.md §13:
  Primary:  MiniLM fine-tuned classifier (minilm_stt_redflag_final)
  Backstop: Aho-Corasick regex/keyword set (runs ALWAYS in parallel)

Both must fail independently for a red flag to be missed.
This is the highest-stakes component — near-zero false negative tolerance.
"""

import time
import re
import os
from typing import Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Model path ────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..")
MINILM_PATH = os.path.join(_BASE, "minilm_stt_redflag_final")

# ── Aho-Corasick Backstop (simplified as compiled regex set) ─────────────────
# This is NOT Aho-Corasick's weakness (single-word keyword exact match).
# These are SEMANTIC CLUSTER patterns — each one is a *category* of expressions
# not a specific phrase. They cover the failure mode: cardiac + neuro emergencies
# in Hinglish including code-switched and STT-garbled variants.
# Per §13: AC runs in parallel, ALWAYS. Not replaced by MiniLM.
_CARDIAC_PATTERNS = [
    r"(chest|seena|chhati|seene|seene\s*mein)\s*(mein\s*)?(dard|pain|pressure|tightness|bhaari|dabaav|jalan)",
    r"(left\s*(haath|hand|arm)|baayan\s*(haath|bazu|taraf))\s*(mein\s*)?(sunn|numb|pain|dard|weakness|kamzori)",
    r"(heart\s*attack|dil\s*ka\s*daura|cardiac|heart\s*fail)",
    r"(thanda\s*paseena|cold\s*sweat|pasiina\s*aa\s*raha)",
    r"(jaw|jada|jaws|jaw\s*pain|jada\s*mein\s*dard)",
]
_NEURO_PATTERNS = [
    r"(stroke|paralysis|laqwa|ek\s*taraf\s*(kamzori|weakness|numb))",
    r"(bolne|baat\s*karne)\s*(mein\s*)?(dikkat|mushkil|ladkhada)",
    r"(aankhon\s*(ke\s*)?(aage\s*)?andhera|vision\s*blur|dhundhla\s*dikh)",
    r"(behoshi|unconscious|hosh\s*kho|gir\s*pad)",
]
_BREATHING_PATTERNS = [
    r"(saans\s*(nahi|nahin|nhi)\s*(aa\s*rahi|le\s*pa)\b)",
    r"(breathless|breathing\s*(problem|difficulty|dikkat|mushkil))",
    r"(ghut\s*raha|suffocating|oxygen\s*nahi)",
]

_ALL_AC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in 
                    _CARDIAC_PATTERNS + _NEURO_PATTERNS + _BREATHING_PATTERNS]


_NEGATION_WINDOW = re.compile(
    r"(nahi|nahin|nhi|no|not|don't|dont)\s+\w{0,8}\s*(hai|tha|thi|hoon|hun|hu|hoga|hogi)\b",
    re.IGNORECASE
)

def _ac_backstop_check(text: str) -> Tuple[bool, str]:
    """
    Always-on parallel check. Returns (fired: bool, matched_pattern: str).
    Must be fast (<1ms) — no model calls.
    
    Applies a negation window: if a negation phrase appears within 30 chars
    AFTER the matched pattern, treat as negated (not a red flag).
    E.g. "mujhe chest pain nahi hai" → negated → no fire.
    This reduces false positives without masking real alerts.
    """
    for pattern in _ALL_AC_PATTERNS:
        m = pattern.search(text)
        if m:
            # Check for negation in a window after the match
            window_start = m.end()
            window_text = text[window_start:window_start + 35]
            if _NEGATION_WINDOW.search(window_text):
                continue  # Negated — skip this match
            return True, m.group(0)
    return False, ""


# ── MiniLM Classifier ─────────────────────────────────────────────────────────
_tokenizer = None
_model = None
_device = None

def _load_minilm():
    global _tokenizer, _model, _device
    if _model is not None:
        return  # already loaded
    t0 = time.perf_counter()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MINILM_PATH)
    _model.eval()
    _model.to(_device)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[SafetyClassifier] MiniLM loaded in {elapsed:.0f}ms on {_device}")


def _minilm_predict(text: str) -> Tuple[float, int]:
    """Returns (red_flag_probability, predicted_class)."""
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=-1)[0]
    pred = int(probs.argmax().item())
    return round(probs[1].item(), 4), pred  # prob of class 1 = red flag


# ── Public API ────────────────────────────────────────────────────────────────

MINILM_THRESHOLD = 0.45  # Fire if p(red_flag) >= 0.45. Lower = higher recall.
                          # Deliberately aggressive — FP cost is low, FN cost is high.

def classify(text: str) -> Dict[str, Any]:
    """
    Run both classifiers in sequence (parallel in prod via threads, sequential here).
    Returns a unified result dict.
    
    A turn is flagged if EITHER fires.
    """
    t_start = time.perf_counter()

    # 1. AC backstop (always, fast)
    ac_fired, ac_matched = _ac_backstop_check(text)
    ac_ms = (time.perf_counter() - t_start) * 1000

    # 2. MiniLM primary
    _load_minilm()
    t_ml = time.perf_counter()
    ml_prob, ml_pred = _minilm_predict(text)
    ml_ms = (time.perf_counter() - t_ml) * 1000

    ml_fired = ml_prob >= MINILM_THRESHOLD

    # Final decision: either classifier fires → flag it
    fired = ac_fired or ml_fired
    
    total_ms = (time.perf_counter() - t_start) * 1000

    return {
        "fired": fired,
        "minilm": {
            "red_flag_probability": ml_prob,
            "predicted_class": ml_pred,
            "fired": ml_fired,
            "threshold": MINILM_THRESHOLD,
            "latency_ms": round(ml_ms, 2),
        },
        "ac_backstop": {
            "fired": ac_fired,
            "matched_pattern": ac_matched,
            "latency_ms": round(ac_ms, 3),
        },
        "total_latency_ms": round(total_ms, 2),
    }


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        ("mujhe bhook lagi", False),
        ("mujhe heart attack aa raha hai", True),
        ("mere seene mein dard hai aur left haath sunn hai", True),
        ("achanak se saans nahi aa rahi hai", True),
        ("workout karte hue seene mein dard ho raha hai", True),
        ("meri bolne ki awaaz ladkhada rahi hai aur ek taraf kamzori hai", True),
        ("seene mein dabav hai aur thanda paseena aa raha hai", True),
        ("kal doctor se milna hai", False),
        ("mujhe chest pain nahi hai", False),
        ("mere papa ko abhi seene mein dard hai", True),
        ("thodi bloating ho rahi hai periods ke time", False),
        ("gym join kiya maine aaj", False),
    ]

    print("\n=== Safety Classifier Self-Test ===\n")
    for text, expected in test_cases:
        result = classify(text)
        status = "OK" if result["fired"] == expected else "WRONG"
        print(f"[{status}] '{text[:55]}'")
        print(f"       MiniLM={result['minilm']['red_flag_probability']:.3f} fired={result['minilm']['fired']} | "
              f"AC fired={result['ac_backstop']['fired']} matched='{result['ac_backstop']['matched_pattern'][:30]}' | "
              f"total={result['total_latency_ms']:.1f}ms")
        print()
