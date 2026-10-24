"""
disha_v2/output_guardrail.py
------------------------------
Class C — Hard behavioral constraints, output-side verification.

Per solution_v2.md §13:
  Output Guardrail: regex pre-filter (no binary classifier yet).
  This runs AFTER dialogue generation, scanning the model's output.

Hard constraint violations (always blocked, independent of persona):
  1. Identity violation — claiming to be human
  2. Capability hallucination — promising WhatsApp/email/external messaging
  3. Medical advice beyond scope — diagnosing conditions, prescribing medications

The guardrail also runs Laya's identity_violation and capability_hallucination
noul checks on the MODEL'S OWN OUTPUT (not user input) — this is the output
guardrail path in the architecture diagram.
"""

import re
import time
import os
from typing import Dict, Any, Tuple

# ── Hard Constraint Patterns (Class C — always in context, persona-independent) ──
# These are NOT keyword matching. Each pattern covers a semantic cluster.

_IDENTITY_VIOLATION_PATTERNS = [
    # Claiming to be human / real person — broad coverage
    r"\b(main|mein|mai)\s+(bhi\s+)?(ek\s+)?(insaan|human|real\s+person|asal\s+insaan|banda|aadmi|aurat|ladki)\s+(hoon|hun|hu|toh)\b",
    r"\b(haan[,.]?\s*)?(main|mein|mai)\s+(bhi\s+)?(insaan|human)\s+(hoon|hun|hu|toh|na)\b",
    r"\b(i\s+am\s+a?\s*(real\s+)?(human|person|doctor|nurse|woman|man))\b",
    r"\b(main\s+AI\s+nahi\s+(hoon|hun|hu))\b",
    r"\b(agar\s+main\s+AI\s+hoti|main\s+sirf\s+AI\s+hoti)\b",
    r"\b(main\s+ek\s+real\s+doctor)\b",
    # Denying AI nature
    r"\b(main\s+koi\s+bot\s+nahi|main\s+robot\s+nahi|main\s+AI\s+nahi)\b",
    r"\b(i('m|\s+am)\s+not\s+a[n]?\s*(bot|robot|AI|machine|algorithm))\b",
]

_CAPABILITY_HALLUCINATION_PATTERNS = [
    # WhatsApp / messaging — broad Hinglish + English coverage
    r"\b(whatsapp\s*(pe|par|mein|ko)\s*(send|bhej|share|dal))\b",
    r"(whatsapp\s+pe\s+.*?send|send\s+.*?whatsapp\s+pe)",
    r"\b(send\s+(kar(ta|ti|unga|ungi|raha|rahi)?\s*(hoon|hun|hu|deti|deta|dunga|dungi)?)\s*(whatsapp|message|msg|sms|email|mail))\b",
    r"\b(main\s+(tumhe?|aapko)\s+(whatsapp|message|email|call)\s+kar(ta|ti|unga|ungi|raha|rahi)?\s*(hoon|hun|hu|deti|deta)?)\b",
    r"\b(whatsapp\s+(bhej|dal|send)\s+d(eta|eti|unga|ungi|eti\s+hoon|eta\s+hoon))\b",
    r"\baapko\s+.*?(whatsapp|message|email)\s+(pe\s+)?(send|bhej|dal|kar)\b",
    r"\bmain\s+(aapko?\s+)?(message|text|whatsapp|email)\s+(kar|bhej|send|dal)\b",
    r"\b(i('ll|\s+will|\s+can)\s+send\s+(you\s+)?(a\s+)?(whatsapp|message|text|email))\b",
    # Making appointments on behalf of user (external action)
    r"\b(appointment\s+book\s+kar(ta|ti|unga|ungi|raha|rahi)?\s*(hoon|hun|hu|deti|deta)?)\b",
    r"\b(i('ll|\s+will)\s+book\s+(the\s+)?appointment)\b",
]

_OUT_OF_SCOPE_MEDICAL_PATTERNS = [
    # Prescribing medications
    r"\b(yeh\s+dawai?\s+lo|take\s+this\s+medicine|isko\s+le\s+lo|tablet\s+le\s+lo|capsule\s+le\s+lo)\b",
    r"\b(main\s+(aapko?\s+)?(diagnose|diagnosis)\s+kar)\b",
    r"\b(aapko\s+[a-z\s]+ki\s+bimari\s+hai)\b",
    r"\b(yeh\s+(cancer|tumor|diabetes|thyroid|bp|heart\s+attack)\s+ka\s+symptom\s+hai)\b",  # diagnosing
]

_ALL_PATTERNS = {
    "identity_violation": [re.compile(p, re.IGNORECASE) for p in _IDENTITY_VIOLATION_PATTERNS],
    "capability_hallucination": [re.compile(p, re.IGNORECASE) for p in _CAPABILITY_HALLUCINATION_PATTERNS],
    "out_of_scope_medical": [re.compile(p, re.IGNORECASE) for p in _OUT_OF_SCOPE_MEDICAL_PATTERNS],
}


def check_output(
    model_output: str,
    laya_scores: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Scan the model's generated response for hard constraint violations.
    
    Args:
        model_output:  The dialogue model's generated text
        laya_scores:   Optional — Laya's per-output scores (identity_violation, 
                       capability_hallucination probabilities) from intent router
                       run on the OUTPUT text (second Laya pass)
    
    Returns:
        passed:       bool — True if no violations found
        violations:   list of detected violation types
        matched:      what triggered each violation
        action:       "pass", "rewrite", or "block"
    """
    t0 = time.perf_counter()
    
    violations = []
    matched_patterns = []
    
    # 1. Regex pass (fast, always runs)
    for violation_type, patterns in _ALL_PATTERNS.items():
        for pattern in patterns:
            m = pattern.search(model_output)
            if m:
                violations.append(violation_type)
                matched_patterns.append({
                    "type": violation_type,
                    "matched": m.group(0),
                    "source": "regex"
                })
                break  # one match per type is enough to flag

    # 2. Laya scores on output (if provided — second Laya pass over model's response)
    if laya_scores:
        if laya_scores.get("identity_violation", 0) > 0.65 and "identity_violation" not in violations:
            violations.append("identity_violation")
            matched_patterns.append({
                "type": "identity_violation",
                "matched": f"Laya p={laya_scores['identity_violation']:.3f}",
                "source": "laya"
            })
        if laya_scores.get("capability_hallucination", 0) > 0.65 and "capability_hallucination" not in violations:
            violations.append("capability_hallucination")
            matched_patterns.append({
                "type": "capability_hallucination",
                "matched": f"Laya p={laya_scores['capability_hallucination']:.3f}",
                "source": "laya"
            })

    # Determine action
    if not violations:
        action = "pass"
    elif "identity_violation" in violations or "capability_hallucination" in violations:
        action = "block"  # these are hard blocks, require rewrite
    else:
        action = "rewrite"  # softer violations — flag but allow edited version

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "passed": len(violations) == 0,
        "action": action,
        "violations": violations,
        "matched_patterns": matched_patterns,
        "latency_ms": round(total_ms, 3),
    }


def get_override_response(violation_type: str, patient_name: str = "aap") -> str:
    """
    Return a safe, pre-approved override response for hard constraint violations.
    These are the fallback responses when the model's output is blocked.
    """
    overrides = {
        "identity_violation": (
            f"Main Disha hoon — {patient_name}ki dedicated AI health coach. "
            "Main ek AI hoon jo aapki health journey mein help karne ke liye bani hoon. "
            "Main insaan nahi hoon, lekin main poori sincerety se aapki help karne ke liye yahaan hoon. "
            "Aap mujh par trust kar sakte hain."
        ),
        "capability_hallucination": (
            "Main directly aapko WhatsApp ya koi message nahi bhej sakti — "
            "main sirf is call mein aapke saath baat kar sakti hoon. "
            "Jo bhi important ho, please app mein note kar lein ya apne coach ko in-app message karein."
        ),
        "out_of_scope_medical": (
            "Yeh ek aisi cheez hai jiske baare mein main seedha koi diagnosis nahi de sakti — "
            "main ek health coaching AI hoon, doctor nahi. "
            "Is baare mein apne doctor se zaroor baat karein — yeh unka kaam hai, mera nahi."
        ),
    }
    return overrides.get(violation_type, "Main is sawaal ka jawaab is waqt nahi de sakti.")


if __name__ == "__main__":
    test_outputs = [
        # Should PASS
        ("Main Disha hoon, aapki AI health coach. Jeera paani bahut helpful hoga.", False),
        ("Haan, aaj diet mein brown rice rakhein.", False),
        # Should FAIL — identity violation
        ("Haan, main bhi insaan hoon na — main samajhti hoon aapko!", True),
        ("Main AI nahi hoon, main real hoon!", True),
        ("I'm not a robot, I genuinely care!", True),
        # Should FAIL — capability hallucination
        ("Main aapko WhatsApp pe diet chart send kar deti hoon.", True),
        ("I'll send you a message on WhatsApp with the details.", True),
        ("Appointment book kar deti hoon main aapke liye abhi.", True),
        # Should FAIL — out of scope medical
        ("Yeh diabetes ka symptom hai aapka.", True),
    ]
    
    print("\n=== Output Guardrail Self-Test ===\n")
    all_correct = True
    for text, should_fail in test_outputs:
        r = check_output(text)
        is_correct = (not r["passed"]) == should_fail
        status = "✓" if is_correct else "✗ WRONG"
        if not is_correct:
            all_correct = False
        print(f"[{status}] Action={r['action']} | Violations={r['violations']} | {r['latency_ms']:.2f}ms")
        print(f"  Text: '{text[:80]}'")
        if r["matched_patterns"]:
            for m in r["matched_patterns"]:
                print(f"  Matched: [{m['type']}] '{m['matched']}'")
        print()
    print(f"\nAll correct: {all_correct}")
