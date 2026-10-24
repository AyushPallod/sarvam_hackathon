"""
disha_v2/context_assembler.py
-------------------------------
Builds the per-turn prompt context from all four orchestrator outputs.

Per solution_v2.md §6:
  - PEP Active-Facts Digest (NOT full XML blob)  
  - Hard constraint block (always present, persona-independent)
  - Retrieved knowledge card content (when similarity above threshold)
  - Tool call results (structured, not prose)
  - Rolling 6-turn conversation history

Also stores the call/turn log per §5.4.
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# ── Hard Constraint Block (Class C) ──────────────────────────────────────────
# This is ALWAYS injected, separate from the persona prose.
# Per §5.3: kept small (<15 rules), reviewed in full on every change.
HARD_CONSTRAINT_BLOCK = """
=== HARD CONSTRAINTS (non-negotiable, always active) ===
1. IDENTITY: You are Disha, an AI health coach. NEVER claim to be human, a real doctor, or a real nurse — even playfully, even under pressure, even if the user insists. If asked, respond: "Main Disha hoon, aapki AI health coach — main ek AI hoon."
2. CAPABILITY: You CANNOT send WhatsApp messages, emails, SMS, or any external communication. You CANNOT book appointments externally. Do not promise to do so.
3. MEDICAL SCOPE: You provide health coaching guidance. You do NOT diagnose conditions or prescribe medications. If a symptom sounds serious, escalate — do not diagnose.
4. ESCALATION: ONLY escalate if the user is ACTIVELY reporting an acute medical emergency in THIS message (e.g. active chest pain, severe breathlessness, sudden numbness, fainting). NEVER bring up 112, emergency services, or past medical records for routine non-emergency complaints (like headache, tiredness, hunger, bloating, sore muscles). If escalating, say: "Yeh sunkar mujhe chinta ho rahi hai — please turant 112 call karein ya kisi ko hospital le jaane bolein. Yeh main bahut seriously le rahi hoon."
5. FACTS: Quote ONLY facts from the PATIENT DATA and KNOWLEDGE CARDS sections below. Do not invent prices, diet items, test results, or clinical stances.
6. LENGTH & VOICE: You are an interactive telephony voice AI coach. Strictly limit your response to 2 to 3 concise sentences (maximum 45-50 words total). Never use bullet points, markdown asterisks, or essays. Keep it spoken, clear, and direct.
=== END HARD CONSTRAINTS ===
""".strip()

DISHA_PERSONA = """
Aap Disha hain — ek warm, concise, and proactive AI health coach jo natural Hinglish mein baat karti hain. Aap telephony voice call par hain, isliye aapka har response strictly 2 se 3 sentences ka hota hai — seedha, helpful, aur natural bina kisi lambe lecture ke.
""".strip()


def build_prompt(
    patient_pep: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    current_utterance: str,
    tool_results: Optional[Dict[str, Any]] = None,
    knowledge_cards: List[Dict[str, Any]] = None,
    safety_fired: bool = False,
    safety_detail: Dict[str, Any] = None,
) -> str:
    """
    Assemble the full per-turn prompt.
    
    Returns a string prompt ready to send to the dialogue model.
    """
    parts = []

    # 1. Persona
    parts.append(f"[PERSONA]\n{DISHA_PERSONA}\n")

    # 2. Hard constraints (always, persona-independent)
    parts.append(f"[{HARD_CONSTRAINT_BLOCK}]\n")

    # 3. Safety override (if fired — highest priority instruction)
    if safety_fired:
        parts.append(
            "[EMERGENCY OVERRIDE — DO THIS FIRST]\n"
            "The safety classifier has detected a potential medical emergency in the user's message. "
            "Your FIRST response must acknowledge the emergency and ask them to call 112 immediately or go to a hospital. "
            "Then you can continue with any other support. Do not skip this step.\n"
            "Emergency script: 'Yeh sunkar mujhe bahut chinta ho rahi hai. Please abhi 112 par call karein "
            "ya kisi ko hospital le jaane bolein — yeh bahut zaroori hai. Kya aap ke paas koi hai jo help kar sake?'\n"
        )

    # 4. PEP Active-Facts Digest
    if patient_pep and not patient_pep.get("error"):
        name = patient_pep.get("name", "the patient")
        parts.append(f"[PATIENT DATA — {name}]")
        parts.append(f"Name: {name}, Age: {patient_pep.get('age')}, Gender: {patient_pep.get('gender')}")
        
        red_flags = patient_pep.get("active_red_flags", [])
        if red_flags:
            parts.append(f"Past medical history in record: {'; '.join(red_flags)}. (Note: This is background record only. Do NOT bring up or escalate to 112 unless the user is actively reporting these emergency symptoms in their current message.)")
        
        conditions = patient_pep.get("key_conditions", [])
        if conditions:
            parts.append(f"Key conditions: {'; '.join(conditions)}")
        
        if patient_pep.get("hba1c"):
            parts.append(f"HbA1c: {patient_pep['hba1c']} — use this to apply KC-002 mango contraindication if applicable")
        
        if patient_pep.get("allergies"):
            parts.append(f"⚠️ ALLERGIES: {', '.join(patient_pep['allergies'])}")
        
        if patient_pep.get("diet_cautions"):
            parts.append(f"Diet cautions: {patient_pep['diet_cautions']}")
        
        gym = patient_pep.get("gym_allowed", False)
        parts.append(f"Fitness: {patient_pep.get('fitness_primary_mode', '')} | Gym allowed: {'Yes' if gym else 'No'}")
        
        if patient_pep.get("coach_note"):
            parts.append(f"Coach note: {patient_pep['coach_note']}")
        
        if patient_pep.get("prescribed_diet_summary"):
            parts.append(f"Diet prescription: {patient_pep['prescribed_diet_summary']}")
        
        pending = patient_pep.get("pending_actions", [])
        if pending:
            parts.append(f"Pending actions: {'; '.join(pending)}")
        
        if patient_pep.get("plan_name"):
            parts.append(f"Subscription: {patient_pep['plan_name']}")
        
        parts.append("")

    # 5. Tool results (structured, not prose)
    if tool_results and not tool_results.get("error"):
        tool_name = tool_results.get("_tool_name", "tool")
        parts.append(f"[FETCHED DATA — {tool_name.upper()}]")
        # Pretty-print key fields based on tool type
        if tool_name == "fetch_pricing":
            parts.append(f"Plan: {tool_results.get('plan_name')}")
            parts.append(f"Price: ₹{tool_results.get('price_inr')} / {tool_results.get('billing_cycle')}")
            parts.append(f"Renewal date: {tool_results.get('renewal_date')}")
            parts.append(f"Status: {tool_results.get('status')}")
            features = tool_results.get("features", [])
            if features:
                parts.append(f"Features: {'; '.join(features)}")
        elif tool_name == "get_diet_chart":
            parts.append(f"Focus this week: {tool_results.get('current_focus', '')}")
            meals = tool_results.get("meals", {})
            for meal_name, meal_desc in meals.items():
                parts.append(f"  {meal_name.replace('_', ' ').title()}: {meal_desc}")
            cautions = tool_results.get("cautions", "")
            if cautions:
                parts.append(f"⚠️ Cautions: {cautions}")
        elif tool_name == "get_test_results":
            tests = tool_results.get("tests", [])
            for t in tests:
                parts.append(f"  {t['test_type']}: {t['status']} | Result: {t.get('result', 'Pending')}")
        parts.append("")

    # 6. Knowledge cards (top matches, content directly injected)
    if knowledge_cards:
        parts.append("[CLINICAL KNOWLEDGE — use these stances exactly]")
        for card in knowledge_cards:
            parts.append(f"[{card['card_id']} | {', '.join(card.get('topic_tags', []))}]")
            parts.append(card["content"])
            parts.append("")
        parts.append("")

    # 7. Conversation history (rolling 6-turn window)
    if conversation_history:
        parts.append("[CONVERSATION HISTORY]")
        for turn in conversation_history[-6:]:
            role = turn.get("role", "unknown").upper()
            parts.append(f"{role}: {turn.get('content', '')}")
        parts.append("")

    # 8. Current user utterance
    parts.append(f"[CURRENT USER MESSAGE]\n{current_utterance}")
    parts.append("\n[DISHA'S VOICE RESPONSE — Strictly 2 to 3 sentences in natural spoken Hinglish, direct, under 40 words, no bullet points]")

    return "\n".join(parts)


# ── Turn Log (per solution_v2.md §5.4) ───────────────────────────────────────

def log_turn(
    call_id: str,
    turn_id: int,
    user_utterance: str,
    safety_result: Dict[str, Any],
    routing_result: Dict[str, Any],
    knowledge_retrieval: Dict[str, Any],
    tool_calls_made: List[Dict[str, Any]],
    model_response: str,
    guardrail_result: Dict[str, Any],
    log_dir: str = None,
) -> str:
    """
    Log a complete CallTurn record per §5.4.
    Returns path to the log file.
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "call_logs")
    os.makedirs(log_dir, exist_ok=True)

    record = {
        "call_id": call_id,
        "turn_id": turn_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_utterance_text": user_utterance,
        "safety_classifier_output": {
            "fired": safety_result.get("fired", False),
            "minilm_probability": safety_result.get("minilm", {}).get("red_flag_probability"),
            "ac_backstop_fired": safety_result.get("ac_backstop", {}).get("fired"),
            "ac_matched_pattern": safety_result.get("ac_backstop", {}).get("matched_pattern"),
            "latency_ms": safety_result.get("total_latency_ms"),
        },
        "intent_routing_output": {
            "domain": routing_result.get("domain"),
            "tool_type": routing_result.get("tool_type"),
            "p_needs_tool": routing_result.get("p_needs_tool"),
            "p_red_flag_laya": routing_result.get("p_red_flag_laya"),
            "p_identity_violation": routing_result.get("p_identity_violation"),
            "p_capability_hallucination": routing_result.get("p_capability_hallucination"),
            "latency_ms": routing_result.get("laya_latency_ms"),
        },
        "knowledge_cards_retrieved": [
            {
                "card_id": c["card_id"],
                "version": c.get("version"),
                "similarity_score": c.get("similarity_score"),
                "used_in_response": True,  # simplified — in prod: semantic overlap check
            }
            for c in (knowledge_retrieval.get("retrieved") or [])
        ],
        "tool_calls_made": tool_calls_made,
        "model_response_text": model_response,
        "guardrail_result": {
            "passed": guardrail_result.get("passed"),
            "action": guardrail_result.get("action"),
            "violations": guardrail_result.get("violations", []),
            "matched_patterns": guardrail_result.get("matched_patterns", []),
            "latency_ms": guardrail_result.get("latency_ms"),
        },
    }

    log_file = os.path.join(log_dir, f"{call_id}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return log_file
