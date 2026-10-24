"""
disha_v2/db.py
--------------
Structured DB layer — the single source of truth.
All tool calls (get_diet_chart, fetch_pricing, get_test_results, get_patient_pep)
hit this module, NOT context blobs.
"""

import json
import os
from datetime import datetime
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Tool Implementations ────────────────────────────────────────────────────

def fetch_pricing(user_id: str) -> dict:
    """Return the exact subscription record — no guessing, no generation."""
    db = _load("structured_db.json")
    rec = db["subscriptions"].get(user_id)
    if not rec:
        return {"error": f"No subscription found for {user_id}"}
    return {
        "user_id": user_id,
        "plan_name": rec["plan_name"],
        "price_inr": rec["price_inr"],
        "billing_cycle": rec["billing_cycle"],
        "renewal_date": rec["renewal_date"],
        "status": rec["status"],
        "features": rec["features"],
    }


def get_diet_chart(user_id: str, day: str = "today") -> dict:
    """Return the structured diet chart for a given patient and day."""
    db = _load("structured_db.json")
    patient = db["patients"].get(user_id)
    if not patient:
        return {"error": f"Patient {user_id} not found"}

    chart_id = patient.get("active_diet_chart_id")
    if not chart_id:
        return {"error": f"No active diet chart for {user_id}"}

    chart = db["diet_charts"].get(chart_id)
    if not chart:
        return {"error": f"Diet chart {chart_id} not found in DB"}

    day_data = chart["days"].get(day) or chart["days"].get("today")
    if not day_data:
        return {"error": f"No diet data for day={day}"}

    return {
        "patient_id": user_id,
        "chart_id": chart_id,
        "week_number": chart["week_number"],
        "current_focus": chart["current_focus"],
        "day": day,
        "meals": day_data,
        "cautions": db["patients"][user_id]["diet_preferences"].get("cautions", ""),
        "allergies": db["patients"][user_id]["medical_profile"].get("allergies", []),
    }


def get_test_results(user_id: str) -> dict:
    """Return all test orders and their statuses for a patient."""
    db = _load("structured_db.json")
    orders = db["test_orders"].get(user_id, [])
    if not orders:
        return {"user_id": user_id, "tests": [], "note": "No tests ordered or on record"}

    return {
        "user_id": user_id,
        "tests": [
            {
                "order_id": o["order_id"],
                "test_type": o["test_type"],
                "status": o["status"],
                "ordered_date": o.get("ordered_date"),
                "result_date": o.get("result_date"),
                "result": o.get("result", "Pending"),
                "lab": o.get("lab", ""),
                "notes": o.get("notes", ""),
            }
            for o in orders
        ],
    }


def get_patient_pep(user_id: str) -> dict:
    """
    Return the PEP Active-Facts Digest (NOT the full XML blob).
    Short, structured extraction of what the dialogue model actually needs:
    - prescribed plan type + key constraints + most recent flags
    Per solution_v2.md §6 Note: digest not full blob.
    """
    db = _load("structured_db.json")
    p = db["patients"].get(user_id)
    if not p:
        return {"error": f"Patient {user_id} not found"}

    sub = db["subscriptions"].get(user_id, {})

    return {
        "patient_id": user_id,
        "name": p["name"],
        "age": p["age"],
        "gender": p["gender"],
        "active_red_flags": p["medical_profile"].get("active_red_flags", []),
        "key_conditions": p["medical_profile"].get("conditions", [])[:3],  # top 3 only
        "hba1c": p["medical_profile"].get("hba1c"),
        "allergies": p["medical_profile"].get("allergies", []),
        "diet_cautions": p["diet_preferences"].get("cautions", ""),
        "fitness_primary_mode": p["fitness_routine"].get("primary_mode", ""),
        "gym_allowed": p["fitness_routine"].get("gym_allowed", False),
        "coach_note": p["fitness_routine"].get("coach_note", ""),
        "prescribed_diet_summary": p["prescribed_interventions"].get("diet", ""),
        "pending_actions": p["prescribed_interventions"].get("pending_actions", []),
        "plan_name": sub.get("plan_name", ""),
    }
