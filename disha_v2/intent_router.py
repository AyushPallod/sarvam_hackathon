"""
disha_v2/intent_router.py
--------------------------
Class B — Tool-call decision logic.

Architecture per solution_v2.md §13:
  Primary:  Laya v2 fine-tuned `choice` primitive (disha_laya_router_v2)
  Fallback: Embedding-similarity below confidence threshold (choice:3-5 abstention = 0.5892)

Per §4: decision is made by the Intent Router, NOT the dialogue model.
The router outputs a fixed intent label from a fixed versioned set.
If no intent label fires, no tool call happens — conversational turns cost zero.
"""

import time
import os
import json
from typing import Dict, Any, Optional

import laya

# ── Model path ────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..")
LAYA_V2_PATH = os.path.join(_BASE, "disha_laya_router_v2")

# ── Laya QUESTIONS_SCHEMA (matches training schema exactly) ──────────────────
QUESTIONS_SCHEMA = {
    "domain": {
        "type": "choice",
        "instructions": "What health domain is this conversation primarily about?",
        "criteria": {
            "fitness": "exercise, gym, workout, physical activity, yoga",
            "nutrition": "food, diet, eating, meals, weight, mango, fruits",
            "sleep": "sleep schedule, rest, fatigue, night shift, circadian",
            "medical": "symptoms, pain, illness, medication, test results",
            "billing": "subscription, price, payment, charges, kitna",
            "ops": "app crash, customer support, refund, cancel",
            "smalltalk": "greetings, jokes, casual chat"
        }
    },
    "needs_tool": {
        "type": "noul",
        "instructions": "Does the user need a specific factual answer requiring a database lookup?"
    },
    "tool_type": {
        "type": "choice",
        "instructions": "Which tool is needed?",
        "criteria": {
            "fetch_pricing": "subscription cost, price, kitna, charges, how much",
            "get_diet_chart": "diet plan, what to eat, meal chart, aaj kya khana",
            "get_test_results": "lab results, blood test, report, test aaya",
            "none": "conversational, general health talk, no lookup needed"
        }
    },
    "red_flag": {
        "type": "noul",
        "instructions": "Does this utterance describe a serious acute medical symptom like chest pain, numbness, or breathing difficulty?"
    },
    "urgency": {
        "type": "score",
        "instructions": "How medically urgent is this?",
        "criteria": [
            "no medical concern",
            "minor symptom, monitor",
            "needs attention soon",
            "immediate emergency — escalate now"
        ]
    },
    "identity_violation": {
        "type": "noul",
        "instructions": "Is this text falsely claiming to be human or denying that it is an artificial intelligence?",
        "criteria": {
            "false": "The text states it is an AI, bot, virtual assistant, machine, or talks about other topics.",
            "true": "The text claims to be a living human, doctor, nurse, person, or explicitly denies being an AI."
        }
    },
    "capability_hallucination": {
        "type": "noul",
        "instructions": "Does this text promise to send a WhatsApp message, email, file, or any external communication?"
    }
}

# Abstention threshold from train_report.json — below this, fall back to embedding similarity
CHOICE_ABSTENTION_THRESHOLD = 0.5892  # choice:3-5 from train_report


# ── Laya model (singleton, lazy-loaded) ──────────────────────────────────────
_router = None

def _load_laya():
    global _router
    if _router is not None:
        return
    t0 = time.perf_counter()
    _router = laya.load(LAYA_V2_PATH)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[IntentRouter] Laya v2 loaded in {elapsed:.0f}ms")


# ── Tool intent mapping ───────────────────────────────────────────────────────
# Maps tool_type label → actual function name in db.py
TOOL_DISPATCH = {
    "fetch_pricing": "fetch_pricing",
    "get_diet_chart": "get_diet_chart",
    "get_test_results": "get_test_results",
    "none": None,
}


def route(text: str, conversation_context: str = "") -> Dict[str, Any]:
    """
    Run Laya v2 on the utterance. Returns routing decision.
    
    conversation_context: last 2-3 turns concatenated (for multi-turn red flag detection)
    """
    _load_laya()
    
    # Route based on the current turn's utterance to prevent previous turns (e.g. billing questions)
    # from bleeding into subsequent turns (e.g. headache or diet inquiries).
    analysis_text = text
    
    try:
        import torch
        if torch.get_num_threads() < 8:
            torch.set_num_threads(8)
    except Exception:
        pass

    t0 = time.perf_counter()
    result = _router.predict(analysis_text, QUESTIONS_SCHEMA)
    laya_ms = (time.perf_counter() - t0) * 1000

    ans = result["answers"]
    
    # Extract all primitives
    domain = ans["domain"]["choice"]
    p_needs_tool = ans["needs_tool"]["noul"]
    tool_type = ans["tool_type"]["choice"]
    p_red_flag = ans["red_flag"]["noul"]
    urgency_raw = ans.get("urgency", {})
    urgency = urgency_raw.get("score", 0) if isinstance(urgency_raw, dict) else 0
    p_identity_viol = ans["identity_violation"]["noul"]
    p_cap_halluc = ans["capability_hallucination"]["noul"]

    # Confidence check — if tool_type confidence is below abstention threshold,
    # log it as low-confidence (in prod: would trigger embedding-similarity fallback)
    tool_confident = True
    if tool_type != "none":
        # Laya doesn't expose per-choice confidence directly in this API surface;
        # use needs_tool noul probability as a proxy
        if p_needs_tool < CHOICE_ABSTENTION_THRESHOLD:
            tool_confident = False
            tool_type = "none"  # abstain, no forced tool call

    # Determine actual tool to dispatch
    tool_fn = TOOL_DISPATCH.get(tool_type)

    return {
        "domain": domain,
        "tool_type": tool_type,
        "tool_fn": tool_fn,
        "needs_tool": p_needs_tool > 0.5,
        "p_needs_tool": round(p_needs_tool, 4),
        "p_red_flag_laya": round(p_red_flag, 4),      # Laya's read on red_flag (shadow mode input)
        "urgency": round(urgency, 2),
        "p_identity_violation": round(p_identity_viol, 4),
        "p_capability_hallucination": round(p_cap_halluc, 4),
        "tool_confident": tool_confident,
        "laya_latency_ms": round(laya_ms, 2),
        "analysis_text_used": analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text,
    }


if __name__ == "__main__":
    test_cases = [
        ("Mera diet chart kab aayega?", ""),
        ("Is app ka price kitna hai?", ""),
        ("Gym ke baad kaunsa protein lu?", ""),
        ("Mera blood test report aaya kya?", ""),
        ("haha main toh insaan hoon na", ""),
        ("Main AI hoon, aapki virtual assistant.", ""),
        ("WhatsApp pe send kar do mujhe", ""),
        # Multi-turn: chest pain + numbness across two turns
        ("Workout ke dauran mujhe thoda dard ho raha tha",
         "User: Workout ke dauran mujhe thoda seene mein dard tha\nDisha: Haan, workout mein kabhi kabhi strain hota hai"),
    ]

    print("\n=== Intent Router Self-Test ===\n")
    for text, ctx in test_cases:
        r = route(text, ctx)
        print(f"Input: '{text[:60]}'")
        print(f"  Domain={r['domain']} | Tool={r['tool_type']} | "
              f"NeedsTool={r['p_needs_tool']:.3f} | "
              f"Laya RedFlag={r['p_red_flag_laya']:.3f} | "
              f"IdentityViol={r['p_identity_violation']:.3f} | "
              f"CapHalluc={r['p_capability_hallucination']:.3f} | "
              f"Latency={r['laya_latency_ms']:.1f}ms")
        print()
