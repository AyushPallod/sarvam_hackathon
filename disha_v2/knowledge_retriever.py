"""
disha_v2/knowledge_retriever.py
---------------------------------
Class A — Knowledge Card retrieval using real sentence-transformer embeddings + Redis.

Architecture per solution_v2.md §5.1:
  - Embed current turn + short window of recent turns
  - Cosine similarity search against knowledge card trigger_description embeddings
  - Redis stores pre-computed embeddings (real Redis via fakeredis for local dev)
  - Returns top-k cards above similarity threshold, filtered to status=live

Redis key schema:
  kc:embed:{card_id}    → JSON float[] (embedding vector)
  kc:meta:{card_id}     → JSON metadata
  kc:index              → JSON list of all card IDs

This is a real embedding pipeline, not dummy vectors.
"""

import time
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple

# ── sentence-transformers for real embeddings ─────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Redis connection (try real Redis first, fall back to fakeredis for local dev) ──
try:
    import redis as redis_lib
    _r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
    _r.ping()
    REDIS_BACKEND = "real"
    print("[KnowledgeRetriever] Connected to real Redis at localhost:6379")
except Exception:
    import fakeredis
    _r = fakeredis.FakeRedis(decode_responses=True)
    REDIS_BACKEND = "fakeredis"
    print("[KnowledgeRetriever] Using fakeredis (no local Redis server found)")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CARDS_FILE = os.path.join(DATA_DIR, "knowledge_cards.json")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual MiniLM — handles Hinglish code-switching without language-specific hacks

SIMILARITY_THRESHOLD = 0.35   # Cards below this score are not injected
TOP_K = 3                     # Max cards to retrieve per turn
PROACTIVE_TOPIC_MATCH = True  # Also fire proactive-offer cards on PEP topic match

_embed_model: SentenceTransformer = None


def _load_embed_model():
    global _embed_model
    if _embed_model is not None:
        return
    t0 = time.perf_counter()
    try:
        _embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    except Exception:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[KnowledgeRetriever] SentenceTransformer loaded in {elapsed:.0f}ms")


def _embed(text: str) -> np.ndarray:
    """Embed a single string. Returns float32 numpy vector."""
    _load_embed_model()
    return _embed_model.encode(text, normalize_embeddings=True).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors (dot product suffices)."""
    return float(np.dot(a, b))


# ── Index building / cache management ────────────────────────────────────────

def index_single_card(card: Dict[str, Any]) -> None:
    """Index or update a single Knowledge Card in Redis immediately."""
    _load_embed_model()
    card_id = card["id"]
    embed_input = card["trigger_description"] + " " + " ".join(card.get("topic_tags", []))
    vec = _embed_model.encode(embed_input, normalize_embeddings=True).astype(np.float32)

    _r.set(f"kc:embed:{card_id}", json.dumps(vec.tolist()))
    meta = {k: v for k, v in card.items() if k != "content"}
    _r.set(f"kc:meta:{card_id}", json.dumps(meta))
    _r.set(f"kc:content:{card_id}", card.get("content", ""))

    # Update index list
    existing = _r.get("kc:index")
    ids = json.loads(existing) if existing else []
    if card_id not in ids:
        ids.append(card_id)
        _r.set("kc:index", json.dumps(ids))


def remove_single_card(card_id: str) -> None:
    """Remove a Knowledge Card from Redis."""
    _r.delete(f"kc:embed:{card_id}")
    _r.delete(f"kc:meta:{card_id}")
    _r.delete(f"kc:content:{card_id}")
    existing = _r.get("kc:index")
    if existing:
        ids = json.loads(existing)
        if card_id in ids:
            ids.remove(card_id)
            _r.set("kc:index", json.dumps(ids))


def build_index(force: bool = False) -> int:
    """
    Load knowledge_cards.json, embed each card's trigger_description,
    store in Redis. Supports 'live', 'canary', and 'shadow' status cards per §7.
    Returns number of cards indexed.
    """
    existing = _r.get("kc:index")
    if existing and not force:
        ids = json.loads(existing)
        print(f"[KnowledgeRetriever] Index already in Redis: {len(ids)} cards ({REDIS_BACKEND})")
        return len(ids)

    _load_embed_model()
    with open(CARDS_FILE, "r", encoding="utf-8") as f:
        cards = json.load(f)

    indexed_ids = []
    t0 = time.perf_counter()
    
    for card in cards:
        status = card.get("status", "live")
        # Index live, canary, and shadow cards (shadow mode needs active retrieval)
        if status not in ["live", "canary", "shadow"]:
            continue

        card_id = card["id"]
        embed_input = card["trigger_description"] + " " + " ".join(card.get("topic_tags", []))
        vec = _embed_model.encode(embed_input, normalize_embeddings=True).astype(np.float32)
        
        _r.set(f"kc:embed:{card_id}", json.dumps(vec.tolist()))
        meta = {k: v for k, v in card.items() if k != "content"}
        _r.set(f"kc:meta:{card_id}", json.dumps(meta))
        _r.set(f"kc:content:{card_id}", card.get("content", ""))
        
        indexed_ids.append(card_id)

    _r.set("kc:index", json.dumps(indexed_ids))
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[KnowledgeRetriever] Indexed {len(indexed_ids)} cards in {elapsed:.0f}ms ({REDIS_BACKEND})")
    return len(indexed_ids)


def retrieve(
    utterance: str,
    recent_turns: List[str] = None,
    patient_topic_tags: List[str] = None,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Retrieve top-k knowledge cards for the current turn.
    
    Per §7.4:
      - 'live' and 'canary' cards are returned in 'retrieved' (injected into prompt).
      - 'shadow' cards are evaluated against the query, but NEVER injected.
        Instead, any shadow card firing above threshold is logged as a counterfactual.
    """
    t_start = time.perf_counter()
    
    # Ensure index exists
    if not _r.exists("kc:index"):
        build_index()
    
    card_ids = json.loads(_r.get("kc:index"))
    
    # Build query: current utterance + up to 2 recent turns
    query_parts = [utterance]
    if recent_turns:
        query_parts = list(recent_turns[-2:]) + query_parts
    query_text = " ".join(query_parts)
    
    t_embed = time.perf_counter()
    query_vec = _embed(query_text)
    embed_ms = (time.perf_counter() - t_embed) * 1000
    
    # Fetch all card embeddings from Redis and score
    t_search = time.perf_counter()
    scores: List[Tuple[str, float]] = []
    
    for card_id in card_ids:
        raw = _r.get(f"kc:embed:{card_id}")
        if not raw:
            continue
        card_vec = np.array(json.loads(raw), dtype=np.float32)
        sim = _cosine_sim(query_vec, card_vec)
        
        # Boost proactive-offer cards if patient topic tags match
        if patient_topic_tags and PROACTIVE_TOPIC_MATCH:
            meta_raw = _r.get(f"kc:meta:{card_id}")
            if meta_raw:
                meta = json.loads(meta_raw)
                card_tags = set(meta.get("topic_tags", []))
                patient_tags = set(patient_topic_tags)
                overlap = len(card_tags & patient_tags)
                if overlap > 0 and meta.get("category") == "proactive-offer":
                    sim += 0.1 * overlap  # proactive boost

        scores.append((card_id, sim))
    
    search_ms = (time.perf_counter() - t_search) * 1000
    
    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    
    retrieved = []
    shadow_matches = []

    for card_id, sim in scores:
        if sim < SIMILARITY_THRESHOLD:
            continue
        
        meta_raw = _r.get(f"kc:meta:{card_id}")
        if not meta_raw:
            continue
        meta = json.loads(meta_raw)
        status = meta.get("status", "live")
        
        card_info = {
            "card_id": card_id,
            "version": meta.get("version"),
            "category": meta.get("category"),
            "status": status,
            "topic_tags": meta.get("topic_tags", []),
            "similarity_score": round(sim, 4),
            "content": _r.get(f"kc:content:{card_id}") or "",
            "author": meta.get("author", ""),
        }

        if status == "shadow":
            # §7.4: NEVER inject into real response! Log counterfactual instead.
            shadow_matches.append(card_info)
            try:
                from disha_v2 import governance
                governance.log_shadow_counterfactual(card_id, utterance, sim)
            except Exception:
                pass
        elif status in ["live", "canary"]:
            if len(retrieved) < top_k:
                retrieved.append(card_info)
    
    total_ms = (time.perf_counter() - t_start) * 1000
    
    return {
        "retrieved": retrieved,
        "shadow_matches": shadow_matches,
        "query_text": query_text[:120],
        "cards_scored": len(scores),
        "redis_backend": REDIS_BACKEND,
        "latency": {
            "embed_ms": round(embed_ms, 2),
            "redis_search_ms": round(search_ms, 2),
            "total_ms": round(total_ms, 2),
        }
    }


if __name__ == "__main__":
    build_index(force=True)
    
    test_queries = [
        ("Thodi bloating ho rahi hai, especially periods ke time", 
         [], ["menstrual-health", "bloating"]),
        ("Gym join kiya maine aaj — kya workout karna chahiye?", 
         [], ["fitness"]),
        ("Is app ka price kitna hai?",
         [], []),
        ("Kya aam kha sakti hoon main? Weight loss pe hoon",
         [], ["nutrition"]),
        ("Raat ko neend nahi aati, night shift hai",
         [], ["sleep", "night-shift"]),
        # Multi-turn: first turn about chest pain context, second about numbness
        ("Haath mein thoda sunn feel ho raha hai",
         ["Workout ke dauran seene mein thoda dard tha aaj"], ["fitness"]),
    ]
    
    print("\n=== Knowledge Retriever Self-Test ===\n")
    for utterance, turns, tags in test_queries:
        r = retrieve(utterance, turns, tags)
        print(f"Query: '{utterance[:60]}'")
        print(f"  Embed={r['latency']['embed_ms']:.1f}ms | "
              f"Redis({r['redis_backend']})={r['latency']['redis_search_ms']:.1f}ms | "
              f"Total={r['latency']['total_ms']:.1f}ms")
        if r["retrieved"]:
            for card in r["retrieved"]:
                print(f"  → {card['card_id']} [{card['category']}] "
                      f"sim={card['similarity_score']:.3f} | {card['topic_tags']}")
                print(f"    Content: {card['content'][:80]}...")
        else:
            print("  → No cards above threshold (conversational turn, no injection needed)")
        print()
