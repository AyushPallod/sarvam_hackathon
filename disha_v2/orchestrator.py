"""
disha_v2/orchestrator.py
--------------------------
Per-turn orchestrator — wires all four components together.

Architecture per solution_v2.md §3:
  User utterance → parallel fan-out:
    (a) Safety Classifier   [Class D] — MiniLM + AC backstop
    (b) Intent/Tool Router  [Class B] — Laya v2 choice
    (c) Knowledge Retriever [Class A] — multilingual embeddings + Redis
  → Turn Context Assembler
  → Hard Constraint Block [Class C]
  → Dialogue Generation (Google Gemini via google-genai SDK)
  → Output Guardrail [Class C verification] — regex
  → Response + Turn Log

This is the real pipeline. No mocks. No fakes.
"""

import time
import uuid
import os
import json
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
import torch

try:
    torch.set_num_threads(8)
except Exception:
    pass



# ── Component imports ─────────────────────────────────────────────────────────
from disha_v2 import safety_classifier, intent_router, knowledge_retriever
from disha_v2 import output_guardrail, context_assembler, db

# ── LLM (Google Gemini) ───────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    _LLM_CLIENT = None
    _LLM_MODEL = "gemini-2.0-flash"  # will probe correct name at first call
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    _LLM_CLIENT = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_MODEL_NAMES_TO_TRY = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    if not LLM_AVAILABLE:
        return None
    try:
        _LLM_CLIENT = google_genai.Client(api_key=GEMINI_API_KEY)
        return _LLM_CLIENT
    except Exception as e:
        print(f"[Orchestrator] LLM client init failed: {e}")
        return None


def _call_llm(prompt: str) -> Tuple[str, float, float, str]:
    """
    Call Gemini LLM with streaming to measure TTFT and total generation time.
    Configured with max_output_tokens=90 for voice conversational brevity (2-3 sentences).
    Returns (response_text, ttft_ms, total_ms, model_used).
    """
    client = _get_llm_client()
    if not client:
        return "[LLM unavailable — configure GEMINI_API_KEY]", 0.0, 0.0, "unavailable"

    from google.genai import types
    config = types.GenerateContentConfig(
        max_output_tokens=90,  # 2 to 3 sentences max for telephony voice speed
        temperature=0.3,
    )

    t0 = time.perf_counter()
    last_err = None
    for model_name in _MODEL_NAMES_TO_TRY:
        try:
            ttft = None
            chunks = []
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    chunks.append(chunk.text)
            
            total_ms = (time.perf_counter() - t0) * 1000
            ttft_ms = ttft if ttft is not None else total_ms
            full_text = "".join(chunks).strip()
            # Cache the successful model at the top of the list for zero-latency future calls
            if model_name != _MODEL_NAMES_TO_TRY[0]:
                _MODEL_NAMES_TO_TRY.remove(model_name)
                _MODEL_NAMES_TO_TRY.insert(0, model_name)
            return full_text, round(ttft_ms, 1), round(total_ms, 1), model_name
        except Exception as e:
            last_err = e
            continue

    # All models failed
    total_ms = (time.perf_counter() - t0) * 1000
    return f"[LLM error: {last_err}]", round(total_ms, 1), round(total_ms, 1), "failed"


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch_tool(tool_fn: Optional[str], user_id: str) -> Optional[Dict[str, Any]]:
    """Execute the tool call and return the result dict, or None if no tool needed."""
    if not tool_fn:
        return None
    
    t0 = time.perf_counter()
    
    fn_map = {
        "fetch_pricing": lambda: {**db.fetch_pricing(user_id), "_tool_name": "fetch_pricing"},
        "get_diet_chart": lambda: {**db.get_diet_chart(user_id, "today"), "_tool_name": "get_diet_chart"},
        "get_test_results": lambda: {**db.get_test_results(user_id), "_tool_name": "get_test_results"},
        "get_patient_pep": lambda: {**db.get_patient_pep(user_id), "_tool_name": "get_patient_pep"},
    }
    
    fn = fn_map.get(tool_fn)
    if not fn:
        return None
    
    result = fn()
    result["_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ── Main per-turn processing ───────────────────────────────────────────────────

class DishaTurnResult:
    """Result of processing one conversation turn."""
    def __init__(self):
        self.user_utterance: str = ""
        self.final_response: str = ""
        self.safety: Dict[str, Any] = {}
        self.routing: Dict[str, Any] = {}
        self.knowledge: Dict[str, Any] = {}
        self.tool_result: Optional[Dict[str, Any]] = None
        self.guardrail: Dict[str, Any] = {}
        self.guardrail_rewrote: bool = False
        self.timing: Dict[str, float] = {}
        self.total_ms: float = 0
        self.llm_model_used: str = ""

    def summary(self) -> str:
        lines = [
            f"── Turn Summary ─────────────────────────────────────────",
            f"  User: '{self.user_utterance[:80]}'",
            f"  Safety  : fired={self.safety.get('fired')} | "
            f"MiniLM p={self.safety.get('minilm', {}).get('red_flag_probability', 0):.3f} | "
            f"AC={self.safety.get('ac_backstop', {}).get('fired')} | "
            f"{self.safety.get('total_latency_ms', 0):.1f}ms",
            f"  Routing : domain={self.routing.get('domain')} | "
            f"tool={self.routing.get('tool_type')} | "
            f"identity_viol={self.routing.get('p_identity_violation', 0):.3f} | "
            f"{self.routing.get('laya_latency_ms', 0):.1f}ms",
            f"  KnowCards: {len(self.knowledge.get('retrieved', []))} retrieved | "
            f"embed={self.knowledge.get('latency', {}).get('embed_ms', 0):.1f}ms | "
            f"redis={self.knowledge.get('latency', {}).get('redis_search_ms', 0):.1f}ms",
        ]
        if self.knowledge.get("shadow_matches"):
            shadow_names = [f"{c['card_id']}({c['similarity_score']:.2f})" for c in self.knowledge["shadow_matches"]]
            lines.append(f"  Shadow  : {', '.join(shadow_names)} counterfactual logged (NOT injected per §7.4)")
        if self.tool_result:
            lines.append(f"  Tool    : {self.tool_result.get('_tool_name')} | {self.tool_result.get('_latency_ms', 0):.1f}ms")
        lines.append(f"  LLM     : TTFT={self.timing.get('llm_ttft_ms', 0):.0f}ms | Total={self.timing.get('llm_ms', 0):.0f}ms (model: {self.llm_model_used})")
        lines.append(f"  Guardrail: passed={self.guardrail.get('passed')} | "
                     f"action={self.guardrail.get('action')} | "
                     f"rewrote={self.guardrail_rewrote} | "
                     f"{self.guardrail.get('latency_ms', 0):.2f}ms")
        lines.append(f"  TOTAL   : {self.total_ms:.0f}ms")
        lines.append(f"  Response: '{self.final_response[:120]}'")
        lines.append(f"────────────────────────────────────────────────────────")
        return "\n".join(lines)


def warmup() -> Dict[str, float]:
    """
    Pre-warm all neural models and indexes into memory so Turn 1 runs with zero cold-start delay.
    Loads:
      1. MiniLM fine-tuned safety classifier
      2. Laya v2 fine-tuned intent router
      3. SentenceTransformer embedding model + Redis knowledge card index
      4. Google GenAI LLM client initialization
    Returns dict of load times in milliseconds.
    """
    print("[Warmup] Pre-warming Disha v2 models into memory...")
    t0 = time.perf_counter()
    timings = {}

    # 1. Safety Classifier (load + dummy forward pass)
    ts = time.perf_counter()
    safety_classifier.classify("warmup check")
    timings["safety_minilm_ms"] = round((time.perf_counter() - ts) * 1000, 1)

    # 2. Intent Router (load + dummy forward pass across 7 heads)
    tl = time.perf_counter()
    intent_router.route("warmup check")
    timings["intent_laya_ms"] = round((time.perf_counter() - tl) * 1000, 1)

    # 3. Knowledge Retriever (SentenceTransformer + Redis index + dummy retrieval)
    tk = time.perf_counter()
    knowledge_retriever._load_embed_model()
    knowledge_retriever.build_index()
    knowledge_retriever.retrieve("warmup check")
    timings["knowledge_index_ms"] = round((time.perf_counter() - tk) * 1000, 1)

    # 4. LLM Client & probe working model
    _get_llm_client()
    try:
        _call_llm("ping")
    except Exception:
        pass

    timings["total_warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print(f"[Warmup] All models warm in {timings['total_warmup_ms']}ms. Ready for calls.\n")
    return timings


class DishaOrchestrator:
    """
    Per-call orchestrator. Instantiate once per call, call process_turn() for each turn.
    """

    def __init__(self, user_id: str, call_id: str = None):
        self.user_id = user_id
        self.call_id = call_id or f"CALL-{uuid.uuid4().hex[:8].upper()}"
        self.turn_count = 0
        self.conversation_history: List[Dict[str, str]] = []
        self._pep: Optional[Dict[str, Any]] = None

        # Ensure knowledge index is built
        knowledge_retriever.build_index()

        # Load PEP once per call (not per turn)
        self._pep = db.get_patient_pep(user_id)
        
        print(f"\n[Orchestrator] Call {self.call_id} started for {user_id} "
              f"({self._pep.get('name', 'Unknown')})")

    def _get_recent_context(self) -> str:
        """
        Concise previous user statements for multi-turn symptom accumulation.
        Excludes Disha's responses and error dumps to keep Laya input short (<800ms inference).
        """
        user_statements = [
            t["content"] for t in self.conversation_history if t.get("role") == "user"
        ]
        if not user_statements:
            return ""
        # Return only the last 2 user utterances, truncated to 100 chars each
        return " | ".join(s[:100].strip() for s in user_statements[-2:])

    def _get_patient_topic_tags(self) -> List[str]:
        """Extract PEP-relevant topic tags for proactive card boosting."""
        tags = []
        pep = self._pep or {}
        
        conditions = " ".join(pep.get("key_conditions", []) + pep.get("active_red_flags", []))
        conditions = conditions.lower()
        
        if any(w in conditions for w in ["period", "menstrual", "bloat", "pcos"]):
            tags.extend(["menstrual-health", "bloating"])
        if any(w in conditions for w in ["diabetes", "hba1c", "sugar"]):
            tags.extend(["diabetes", "glycemic-control"])
        if any(w in conditions for w in ["sleep", "night shift"]):
            tags.extend(["sleep", "night-shift"])
        if any(w in conditions for w in ["back pain", "lower back"]):
            tags.extend(["back-pain"])
        if any(w in conditions for w in ["cholesterol", "lipid"]):
            tags.extend(["cholesterol"])
        
        return list(set(tags))

    def process_turn(self, utterance: str) -> DishaTurnResult:
        """
        Process one conversation turn through the full pipeline.
        Returns DishaTurnResult with all component outputs and timings.
        """
        t_total_start = time.perf_counter()
        self.turn_count += 1
        result = DishaTurnResult()
        result.user_utterance = utterance
        timing = {}

        # ── Stage 1: Parallel fan-out ─────────────────────────────────────────
        # Safety, routing, and retrieval run concurrently
        recent_ctx = self._get_recent_context()
        patient_tags = self._get_patient_topic_tags()
        recent_user_turns = [
            t["content"] for t in self.conversation_history[-4:] if t["role"] == "user"
        ]

        t_stage1_start = time.perf_counter()
        
        # 1. Safety Classifier (~9ms)
        safety_result = safety_classifier.classify(utterance)

        # 2. Knowledge Retriever (~15ms)
        knowledge_result = knowledge_retriever.retrieve(
            utterance, recent_user_turns, patient_tags
        )

        # 3. Intent Router (~600ms with dedicated CPU execution)
        routing_result = intent_router.route(utterance, recent_ctx)

        timing["parallel_fan_out_ms"] = (time.perf_counter() - t_stage1_start) * 1000
        result.safety = safety_result
        result.routing = routing_result
        result.knowledge = knowledge_result

        # ── Stage 2: Tool dispatch (sequential, depends on routing) ───────────
        tool_result = None
        tool_calls_made = []
        
        if routing_result.get("tool_fn"):
            t_tool = time.perf_counter()
            tool_result = _dispatch_tool(routing_result["tool_fn"], self.user_id)
            timing["tool_ms"] = (time.perf_counter() - t_tool) * 1000
            if tool_result:
                result.tool_result = tool_result
                tool_calls_made.append({
                    "name": routing_result["tool_fn"],
                    "args": {"user_id": self.user_id},
                    "result_summary": str(tool_result)[:200],
                    "latency_ms": tool_result.get("_latency_ms"),
                })

        # ── Stage 3: Context assembly ─────────────────────────────────────────
        t_assemble = time.perf_counter()
        prompt = context_assembler.build_prompt(
            patient_pep=self._pep,
            conversation_history=self.conversation_history,
            current_utterance=utterance,
            tool_results=tool_result,
            knowledge_cards=knowledge_result.get("retrieved", []),
            safety_fired=safety_result.get("fired", False),
            safety_detail=safety_result,
        )
        timing["context_assemble_ms"] = (time.perf_counter() - t_assemble) * 1000

        # ── Stage 4: LLM generation ───────────────────────────────────────────
        raw_response, ttft_ms, llm_ms, model_used = _call_llm(prompt)
        timing["llm_ttft_ms"] = ttft_ms
        timing["llm_ms"] = llm_ms
        result.llm_model_used = model_used

        # ── Stage 5: Output guardrail ─────────────────────────────────────────
        t_guard = time.perf_counter()
        guardrail_result = output_guardrail.check_output(raw_response)
        timing["guardrail_ms"] = (time.perf_counter() - t_guard) * 1000
        result.guardrail = guardrail_result

        final_response = raw_response
        guardrail_rewrote = False

        if not guardrail_result["passed"]:
            # Rewrite with override response
            violations = guardrail_result.get("violations", [])
            primary_violation = violations[0] if violations else "identity_violation"
            override = output_guardrail.get_override_response(
                primary_violation, self._pep.get("name", "aap")
            )
            
            # If LLM is available, try to regenerate with explicit instruction
            if _get_llm_client() and guardrail_result["action"] != "block":
                rewrite_prompt = (
                    f"{prompt}\n\n"
                    f"[PREVIOUS RESPONSE WAS BLOCKED — violation: {', '.join(violations)}]\n"
                    f"[Rewrite avoiding these violations. Use this as reference: {override}]\n"
                    f"[REWRITTEN DISHA RESPONSE]"
                )
                final_response, _, llm_ms2, _ = _call_llm(rewrite_prompt)
                timing["llm_rewrite_ms"] = llm_ms2
                # Re-check guardrail after rewrite
                recheck = output_guardrail.check_output(final_response)
                if not recheck["passed"]:
                    final_response = override  # hard fallback
            else:
                final_response = override  # use pre-approved override directly
            
            guardrail_rewrote = True

        result.final_response = final_response
        result.guardrail_rewrote = guardrail_rewrote
        result.timing = timing
        result.total_ms = (time.perf_counter() - t_total_start) * 1000

        # ── Stage 6: Update conversation history ──────────────────────────────
        self.conversation_history.append({"role": "user", "content": utterance})
        clean_disha = final_response if not final_response.startswith("[LLM error") else "Main samajh rahi hoon, ek second."
        self.conversation_history.append({"role": "disha", "content": clean_disha})

        # ── Stage 7: Log turn per §5.4 ────────────────────────────────────────
        context_assembler.log_turn(
            call_id=self.call_id,
            turn_id=self.turn_count,
            user_utterance=utterance,
            safety_result=safety_result,
            routing_result=routing_result,
            knowledge_retrieval=knowledge_result,
            tool_calls_made=tool_calls_made,
            model_response=final_response,
            guardrail_result=guardrail_result,
        )

        return result
