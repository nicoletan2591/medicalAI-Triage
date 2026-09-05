"""
triage_agent.py

Main entry point tying together Phase I (RAG retrieval) and Phase II
(dual-layer safety, constrained dialogue) into the full agent.

Flow for every user message:
    1. dual_layer_safety.safety_check_tier() -- ALWAYS runs first
       -> if tier is 'emergency' or 'urgent': return the fixed response
          immediately, skip everything below entirely. The generative
          model is never even called in this path -- true for BOTH
          non-routine tiers now, not just emergency.
    2. If tier is 'routine': retrieve RAG context (Phase I pipeline)
    3. Build a constrained prompt (symptom-mapping / triage-category
       only, per prompt_templates.py)
    4. Dialogue manager decides: ask another clarifying question, or
       assign a final triage category (simple turn-count heuristic --
       see DialogueManager below)
    5. Generate with the local LLM, constrained by the prompt template

CHANGE FROM PRIOR (BINARY) VERSION:
Calls safety_check_tier() instead of safety_check(), and branches on
the three-way tier instead of the boolean escalate flag. The
short-circuit principle is unchanged and now applies to BOTH urgent
and emergency, not just emergency -- dual_layer_safety.py already
returns a fixed, non-generative response for either tier, so
triage_agent.py's job is just to check "is tier routine or not,"
same shape as before, just with a third value.

Every return dict now includes a top-level `tier` field ('emergency',
'urgent', or 'routine') -- this is what should be used to populate
the `predicted_tier` column for triage_metrics.py evaluation, since
it's the single field that's always present regardless of which
branch was taken. `escalated` is kept for backward compatibility
(True for either non-routine tier) but `tier` is more informative and
is the field to use going forward.

Note on the 'routine' + mode='assign' case: once the dialogue manager
decides to assign a final category (after enough clarifying turns),
the actual category text comes from the LLM's generated response
(governed by prompt_templates.py, not shown here), not from `tier`.
`tier` in that case reflects only that the SAFETY layer cleared it as
non-emergency/non-urgent at this turn -- it is not the same thing as
the final assigned triage category. If you need the final assigned
category as part of `predicted_tier` for multi-turn conversations,
that needs to be parsed out of the mode='assign' response text
separately; this file doesn't do that parsing (prompt_templates.py
would need to be involved, and wasn't shared).

This file assumes chroma_db already exists (built by Phase I's
build_index.py) in the same project folder.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

from dual_layer_safety import safety_check_tier
from prompt_templates import build_triage_prompt

GENERATION_MODEL = "qwen2.5"  # your Phase I best-performing model
MIN_CLARIFYING_TURNS = 2   # ask at least this many questions before assigning a category
MAX_CLARIFYING_TURNS = 5   # force a category assignment after this many, to avoid endless questioning


class DialogueManager:
    """Tracks conversation state for a single triage session and decides
    when to keep asking clarifying questions versus assign a final
    category. This is a deliberately simple turn-count heuristic, not
    a learned policy -- consistent with the methodology's requirement
    that dialogue management enforce structured behavior rather than
    being left to open-ended model judgment."""

    def __init__(self):
        self.history = []
        self.clarifying_turns = 0

    def add_turn(self, role, content):
        self.history.append({"role": role, "content": content})

    def should_assign_category(self):
        if self.clarifying_turns >= MAX_CLARIFYING_TURNS:
            return True
        if self.clarifying_turns >= MIN_CLARIFYING_TURNS:
            # In a full implementation, this could check whether enough
            # symptom-mapping fields (onset, severity, associated
            # symptoms) have been covered. Kept as a simple turn
            # threshold here since that logic depends on parsing model
            # output, which should be tested separately before relying
            # on it for this decision.
            return False
        return False


def retrieve_context(query, chroma_path="./chroma_db", top_k=3):
    """Phase I's validated retrieval settings: chunk_size=600 in the
    index (set at build time), top_k=3 (validated in the gold-standard
    sweep as tied-for-best, see tune_rag_params.py results)."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("triage_docs")

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0] if results["documents"][0] else []


def handle_message(user_message, dialogue_manager, use_semantic_safety_layer=True):
    """Processes one user message through the full pipeline. Returns
    the agent's response text and the safety-layer tier for this turn.

    dialogue_manager: a DialogueManager instance, persisted across
    turns by the caller (e.g. one per conversation session)."""

    dialogue_manager.add_turn("user", user_message)

    # Step 1: safety check ALWAYS runs first, before anything else
    safety_result = safety_check_tier(user_message, use_semantic_layer=use_semantic_safety_layer)
    tier = safety_result["tier"]

    if tier != "routine":
        # Fixed response for BOTH urgent and emergency -- generative
        # model is not called for either. safety_check_tier() already
        # picked the right fixed response text for whichever tier this is.
        response = safety_result["response"]
        dialogue_manager.add_turn("assistant", response)
        return {
            "response": response,
            "tier": tier,                # 'emergency' or 'urgent' -- use this for predicted_tier
            "escalated": True,           # backward-compat: True for either non-routine tier
            "safety_detail": safety_result,
        }

    # Step 2: routine per the safety layer -- proceed to RAG-grounded,
    # constrained dialogue. Note this does NOT mean "final category is
    # routine" -- see module docstring note on mode='assign'.
    retrieved_context = retrieve_context(user_message)

    mode = "assign" if dialogue_manager.should_assign_category() else "clarify"
    prompt = build_triage_prompt(retrieved_context, dialogue_manager.history, mode=mode)

    response = ollama.chat(model=GENERATION_MODEL, messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"]

    if mode == "clarify":
        dialogue_manager.clarifying_turns += 1

    dialogue_manager.add_turn("assistant", response_text)

    return {
        "response": response_text,
        "tier": "routine",            # safety-layer tier only -- see docstring re: mode='assign'
        "escalated": False,
        "mode": mode,
        "retrieved_context": retrieved_context,
        "safety_detail": safety_result,
    }
