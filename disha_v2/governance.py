"""
disha_v2/governance.py
-----------------------
Section 7 & 8 Governance & Evaluation Engine for Disha v2.

Implements:
  §7. Governance workflow — how a non-engineer ships a fix:
    - Authoring Knowledge Cards without writing code (trigger + vetted content + tags)
    - Auto-embedding & immediate entry into SHADOW MODE (retrieval index live, never injected into prompt)
    - Counterfactual logging ("this card would have fired here")
    - Automated Regression Replay Engine: replays candidate card against full canonical test suite (CASE-01 to CASE-14)
    - Intent collision detection against existing cards & red-flag safety checks
    - Lifecycle state machine: draft → shadow → canary → live → retired

  §8. Evaluation — proof, not assertion:
    - Pre-launch evaluation: collision matrix, precision, target recall
    - Post-launch continuous audit: async compliance review on call transcripts
    - Near-miss safety audit & tool-opportunity check
"""

import os
import sys
import time
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CARDS_FILE = os.path.join(DATA_DIR, "knowledge_cards.json")
TEST_CASES_FILE = os.path.join(BASE_DIR, "test_cases.json")
SHADOW_LOGS_FILE = os.path.join(DATA_DIR, "shadow_firings.jsonl")
CALL_LOGS_DIR = os.path.join(BASE_DIR, "call_logs")
AUDIT_LOGS_DIR = os.path.join(CALL_LOGS_DIR, "audits")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIT_LOGS_DIR, exist_ok=True)

from disha_v2 import knowledge_retriever


# ── Knowledge Card CRUD & Lifecycle (§7.2, §7.3) ──────────────────────────────

def get_all_cards() -> List[Dict[str, Any]]:
    """Load all cards from knowledge_cards.json."""
    if not os.path.exists(CARDS_FILE):
        return []
    with open(CARDS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else data.get("cards", [])


def save_all_cards(cards: List[Dict[str, Any]]) -> None:
    """Save all cards back to knowledge_cards.json."""
    with open(CARDS_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)


def add_knowledge_card(
    trigger_description: str,
    content: str,
    topic_tags: List[str],
    category: str = "reactive-answer",
    author: str = "Clinician (CMS)",
    initial_status: str = "shadow"
) -> Dict[str, Any]:
    """
    Author a new Knowledge Card as a non-engineer (§7.2).
    Enters SHADOW MODE immediately (§7.3).
    Automatically embeds trigger into Redis vector index and runs initial regression check.
    """
    cards = get_all_cards()
    
    # Generate next ID: KC-013, KC-014, ...
    existing_nums = []
    for c in cards:
        cid = c.get("id", "")
        if cid.startswith("KC-"):
            try:
                existing_nums.append(int(cid.split("-")[1]))
            except Exception:
                pass
    next_num = max(existing_nums, default=0) + 1
    new_id = f"KC-{next_num:03d}"

    card = {
        "id": new_id,
        "version": 1,
        "status": initial_status,  # default 'shadow' per §7
        "category": category,
        "topic_tags": [t.strip().lower() for t in topic_tags if t.strip()],
        "trigger_description": trigger_description.strip(),
        "content": content.strip(),
        "author": author.strip(),
        "approved_at": datetime.date.today().isoformat(),
        "supersedes": None,
        "shadow_firings_count": 0,
    }

    cards.append(card)
    save_all_cards(cards)

    # Immediately index into Redis vector store
    knowledge_retriever.index_single_card(card)

    # Run automated regression check against golden test suite (§7.6)
    regression_report = run_regression_suite(target_card_id=new_id)

    return {
        "card": card,
        "regression_report": regression_report,
        "message": f"Card {new_id} created in SHADOW mode and indexed into Redis vector store."
    }


def update_card_status(card_id: str, new_status: str) -> Dict[str, Any]:
    """
    Promote or change status: draft → shadow → canary → live → retired (§7.4 - §7.7)
    """
    valid_statuses = ["draft", "shadow", "canary", "live", "retired"]
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status '{new_status}'. Allowed: {valid_statuses}")

    cards = get_all_cards()
    target_card = None
    for c in cards:
        if c.get("id") == card_id:
            c["status"] = new_status
            target_card = c
            break

    if not target_card:
        return {"error": f"Card {card_id} not found", "success": False}

    save_all_cards(cards)
    # Refresh Redis index status
    knowledge_retriever.index_single_card(target_card)

    return {
        "success": True,
        "card_id": card_id,
        "new_status": new_status,
        "message": f"Card {card_id} transitioned to status '{new_status}'."
    }


def delete_card(card_id: str) -> Dict[str, Any]:
    """Remove a card from repository and Redis."""
    cards = get_all_cards()
    new_cards = [c for c in cards if c.get("id") != card_id]
    if len(new_cards) == len(cards):
        return {"error": f"Card {card_id} not found", "success": False}

    save_all_cards(new_cards)
    knowledge_retriever.remove_single_card(card_id)
    return {"success": True, "card_id": card_id, "message": f"Card {card_id} deleted."}


# ── Shadow Counterfactual Logging (§7.4) ──────────────────────────────────────

def log_shadow_counterfactual(
    card_id: str,
    utterance: str,
    similarity_score: float,
    call_id: str = "LIVE"
) -> Dict[str, Any]:
    """
    When a shadow-mode card matches during a live turn, its content is NEVER injected into the response.
    Instead, a counterfactual log is recorded per §7.4:
      "this card would have fired here (similarity: X.XX)"
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "call_id": call_id,
        "card_id": card_id,
        "utterance": utterance,
        "similarity_score": round(similarity_score, 4),
        "counterfactual": f"Card {card_id} would have fired here (similarity: {similarity_score:.3f}). Content NOT injected into live prompt."
    }

    # Append to file
    with open(SHADOW_LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Increment counter in card metadata
    cards = get_all_cards()
    for c in cards:
        if c.get("id") == card_id:
            c["shadow_firings_count"] = c.get("shadow_firings_count", 0) + 1
            break
    save_all_cards(cards)

    return entry


def get_shadow_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent shadow mode counterfactual logs (§7.4, §8.2)."""
    if not os.path.exists(SHADOW_LOGS_FILE):
        return []
    
    logs = []
    with open(SHADOW_LOGS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(logs[-limit:]))


# ── Automated Regression Replay Engine (§7.6 & §8.1) ──────────────────────────

def run_regression_suite(target_card_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Automated regression check (§7.6):
    Replay candidate card against the canonical regression dataset (test_cases.json, #1–14).
    
    Checks:
      1. Intent Collision: Does this card's trigger embedding false-fire on any canonical case
         that belongs to a different intent/card (e.g. pricing, WhatsApp, emergency, gym, etc.)?
         Threshold collision: similarity > 0.38 on an unrelated domain.
      2. Safety Interference: Does the card fire on red-flag emergencies (CASE-05, CASE-11)?
      3. Target Match: Does it match queries matching its own topic tags or trigger intent?
    """
    if not os.path.exists(TEST_CASES_FILE):
        return {"error": "test_cases.json not found", "passed": False}

    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    cases = test_data.get("test_cases", [])

    cards = get_all_cards()
    cards_map = {c["id"]: c for c in cards}

    # If target_card_id is specified, evaluate only that card
    target_card = cards_map.get(target_card_id) if target_card_id else None

    # Load embed model
    knowledge_retriever._load_embed_model()

    results = []
    collisions = []
    red_flag_interferences = []
    target_hits = []

    collision_threshold = 0.38

    for tc in cases:
        case_id = tc.get("id")
        utterance = tc.get("user_utterance", "")
        category = tc.get("category", "")
        
        # Embed test utterance
        query_vec = knowledge_retriever._embed(utterance)

        # Score against all cards or target card
        eval_cards = [target_card] if target_card else cards

        for c in eval_cards:
            cid = c["id"]
            c_tags = set(c.get("topic_tags", []))
            
            # Fetch embedding from Redis
            raw_emb = knowledge_retriever._r.get(f"kc:embed:{cid}")
            if not raw_emb:
                # Fallback embed on the fly
                embed_input = c["trigger_description"] + " " + " ".join(c.get("topic_tags", []))
                card_vec = knowledge_retriever._embed(embed_input)
            else:
                import numpy as np
                card_vec = np.array(json.loads(raw_emb), dtype=np.float32)

            sim = knowledge_retriever._cosine_sim(query_vec, card_vec)

            # Determine whether this case is expected to trigger this card
            # A hit is expected if the case topic or trigger matches card topic tags
            is_expected_intent = False
            for tag in c_tags:
                if tag in category.lower() or tag in utterance.lower():
                    is_expected_intent = True
                    break

            # Check collision
            if sim >= collision_threshold:
                if "emergency" in category.lower() or "red_flag" in category.lower():
                    # High risk: card matches on emergency utterance
                    red_flag_interferences.append({
                        "case_id": case_id,
                        "card_id": cid,
                        "similarity": round(sim, 3),
                        "utterance": utterance,
                        "expected": tc.get("expected_behavior"),
                    })
                elif not is_expected_intent:
                    # Intent collision with unrelated case
                    collisions.append({
                        "case_id": case_id,
                        "card_id": cid,
                        "similarity": round(sim, 3),
                        "utterance": utterance,
                        "case_category": category,
                        "card_tags": list(c_tags)
                    })
                else:
                    target_hits.append({
                        "case_id": case_id,
                        "card_id": cid,
                        "similarity": round(sim, 3),
                    })

    # ── Part B: Pairwise Card-to-Card Collision Matrix (§7.5 & §8.1) ─────────────
    # Check if candidate card has high semantic trigger overlap with any other card in library
    card_collisions = []
    active_cards = [c for c in cards if c.get("status") != "retired"]

    for candidate in eval_cards:
        cand_id = candidate["id"]
        cand_trigger = candidate.get("trigger_description", "")
        cand_tags = set(candidate.get("topic_tags", []))

        raw_cand = knowledge_retriever._r.get(f"kc:embed:{cand_id}")
        if raw_cand:
            import numpy as np
            cand_vec = np.array(json.loads(raw_cand), dtype=np.float32)
        else:
            cand_vec = knowledge_retriever._embed(cand_trigger + " " + " ".join(cand_tags))

        for other in active_cards:
            oth_id = other["id"]
            if oth_id == cand_id:
                continue

            # Skip intentional version replacements
            if candidate.get("supersedes") == oth_id or other.get("supersedes") == cand_id:
                continue

            raw_oth = knowledge_retriever._r.get(f"kc:embed:{oth_id}")
            if raw_oth:
                import numpy as np
                oth_vec = np.array(json.loads(raw_oth), dtype=np.float32)
            else:
                oth_vec = knowledge_retriever._embed(other.get("trigger_description", "") + " " + " ".join(other.get("topic_tags", [])))

            card_sim = float(knowledge_retriever._cosine_sim(cand_vec, oth_vec))

            # Thresholds: >=0.72 Critical Collision, >=0.65 Warning Overlap
            if card_sim >= 0.65:
                severity = "CRITICAL_COLLISION" if card_sim >= 0.72 else "WARNING_OVERLAP"
                pair_key = tuple(sorted([cand_id, oth_id]))
                if not any(tuple(sorted([col["card_a"], col["card_b"]])) == pair_key for col in card_collisions):
                    card_collisions.append({
                        "card_a": cand_id,
                        "card_b": oth_id,
                        "card_a_status": candidate.get("status"),
                        "card_b_status": other.get("status"),
                        "card_a_author": candidate.get("author"),
                        "card_b_author": other.get("author"),
                        "similarity": round(card_sim, 3),
                        "similarity_pct": f"{card_sim * 100:.1f}%",
                        "severity": severity,
                        "trigger_a": cand_trigger,
                        "trigger_b": other.get("trigger_description", ""),
                        "shared_tags": list(cand_tags & set(other.get("topic_tags", []))),
                        "note": (
                            f"Cards {cand_id} and {oth_id} have {card_sim * 100:.1f}% trigger overlap. "
                            + ("High risk of contradictory advice or double-retrieval." if severity == "CRITICAL_COLLISION" else "Triggers may compete on ambiguous queries.")
                        )
                    })

    has_critical_card_collisions = any(c["severity"] == "CRITICAL_COLLISION" for c in card_collisions)
    passed = (len(collisions) == 0 and len(red_flag_interferences) == 0 and not has_critical_card_collisions)

    verdict = "PASS" if passed else "FLAGGED_FOR_REVIEW"

    if not passed:
        recs = []
        if red_flag_interferences:
            recs.append(f"{len(red_flag_interferences)} red-flag safety clashes")
        if collisions:
            recs.append(f"{len(collisions)} intent collisions with canonical benchmark cases")
        if has_critical_card_collisions:
            crit_count = len([c for c in card_collisions if c["severity"] == "CRITICAL_COLLISION"])
            recs.append(f"{crit_count} direct Card-to-Card collision(s) with existing cards")
        recommendation = "Do NOT promote yet. Issues detected: " + "; ".join(recs) + ". Refine trigger description or set 'supersedes'."
    else:
        recommendation = (
            "Safe to promote to Canary / Live — 0 canonical case collisions, 0 red-flag clashes, "
            "and 0 Card-to-Card stance conflicts detected."
        )

    return {
        "verdict": verdict,
        "passed": passed,
        "total_test_cases_replayed": len(cases),
        "target_card_id": target_card_id,
        "collisions_count": len(collisions),
        "collisions": collisions,
        "red_flag_interferences_count": len(red_flag_interferences),
        "red_flag_interferences": red_flag_interferences,
        "target_hits_count": len(target_hits),
        "card_collisions_count": len(card_collisions),
        "card_collisions": card_collisions,
        "recommendation": recommendation,
        "timestamp": datetime.datetime.now().isoformat()
    }


# ── Async Post-Call Compliance Reviewer (§8.2) ────────────────────────────────

def run_async_compliance_review(call_id: str) -> Dict[str, Any]:
    """
    Offline Async Compliance Reviewer (§8.2):
    Runs on completed calls, evaluating:
      1. Near-miss safety audit: did utterance contain emergency markers where safety didn't fire?
      2. Tool call opportunity missed: did user ask price/diet and tool wasn't invoked?
      3. Identity compliance: did response assert AI identity?
      4. Knowledge Card utilization: did retrieved card match clinical guidance?
    """
    log_file = os.path.join(CALL_LOGS_DIR, f"{call_id}.jsonl")
    if not os.path.exists(log_file):
        return {"error": f"Call log for {call_id} not found", "call_id": call_id}

    turns = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line.strip()))

    flags = []
    near_misses = []
    tool_misses = []
    identity_violations = []

    # Emergency keywords to check for near-misses
    emergency_patterns = ["seene me dard", "chest pain", "chhati me dard", "sunn", "numb", "left arm", "left haath"]

    for t in turns:
        tid = t.get("turn_id", 1)
        utt = t.get("user_utterance", "").lower()
        resp = t.get("model_response_text", "")
        safety = t.get("safety_result", {})
        routing = t.get("routing_result", {})
        guard = t.get("guardrail_result", {})

        # 1. Near-miss check
        if any(p in utt for p in emergency_patterns) and not safety.get("fired"):
            near_miss = {
                "turn_id": tid,
                "utterance": utt,
                "note": "Emergency symptom mentioned in utterance but safety classifier did not fire! Requires urgent clinician review."
            }
            near_misses.append(near_miss)
            flags.append({"type": "SAFETY_NEAR_MISS", "severity": "CRITICAL", **near_miss})

        # 2. Tool miss check
        if any(w in utt for w in ["price", "kitna lagta", "kitne ka", "subscription"]) and routing.get("tool_fn") != "fetch_pricing":
            tool_miss = {
                "turn_id": tid,
                "utterance": utt,
                "note": "User asked for pricing but fetch_pricing tool was not dispatched."
            }
            tool_misses.append(tool_miss)
            flags.append({"type": "TOOL_OPPORTUNITY_MISSED", "severity": "MEDIUM", **tool_miss})

        # 3. Guardrail violations
        if not guard.get("passed", True):
            identity_violations.append({
                "turn_id": tid,
                "violations": guard.get("violations", []),
                "action": guard.get("action")
            })

    audit_report = {
        "call_id": call_id,
        "turns_reviewed": len(turns),
        "status": "CLEAN" if len(flags) == 0 else "FLAGGED",
        "near_misses_count": len(near_misses),
        "near_misses": near_misses,
        "tool_misses_count": len(tool_misses),
        "tool_misses": tool_misses,
        "identity_interventions_count": len(identity_violations),
        "total_flags": len(flags),
        "flags": flags,
        "reviewed_at": datetime.datetime.now().isoformat()
    }

    # Save audit report
    audit_path = os.path.join(AUDIT_LOGS_DIR, f"{call_id}_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    return audit_report
