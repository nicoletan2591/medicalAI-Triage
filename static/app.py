"""
FastAPI backend for the Local, Offline Conversational Triage and
Symptom-Mapping Agent chat GUI.

UPDATE: this now imports and calls your REAL triage_agent.py directly
-- no more placeholder functions. Place this file in the same folder
as triage_agent.py, dual_layer_safety.py, semantic_intent_classifier.py,
keyword_redflag_layer.py, prompt_templates.py, and your chroma_db/.

Run from that folder:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import uuid
import time

from triage_agent import DialogueManager, handle_message

app = FastAPI(title="Local Triage Agent API")

# Frontend and backend both run on localhost -- CORS is permissive here
# only because nothing ever leaves the machine. Do not deploy this
# configuration anywhere reachable off-device.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store -- each session holds a real DialogueManager
# instance from your triage_agent.py, so conversation state (history,
# clarifying_turns) persists across turns exactly as it would in a
# CLI/notebook version of the same agent.
SESSIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    tier: Literal["emergency", "urgent", "routine"]
    text: str
    turn: int
    mode: Optional[str] = None            # 'clarify' or 'assign', only meaningful when tier='routine'
    generated: bool                       # False for emergency/urgent (fixed responses), True for routine
    confidence: Optional[float] = None    # 0-1, only populated if the pipeline reports one
    keyword_triggered: Optional[bool] = None   # True/False if dual_layer_safety reports which layer fired
    semantic_triggered: Optional[bool] = None  # None for both = pipeline doesn't expose per-layer detail yet


def get_or_create_session(session_id: Optional[str]) -> tuple[str, dict]:
    if session_id and session_id in SESSIONS:
        return session_id, SESSIONS[session_id]
    new_id = session_id or str(uuid.uuid4())
    SESSIONS[new_id] = {
        "dialogue_manager": DialogueManager(),
        "turn": 0,
        "created": time.time(),
    }
    return new_id, SESSIONS[new_id]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    session_id, state = get_or_create_session(req.session_id)
    state["turn"] += 1

    # This one call runs your ENTIRE real pipeline: dual-layer safety
    # check first (keyword + semantic layers), then -- only if tier is
    # 'routine' -- RAG retrieval and constrained generation. The API
    # layer does not reimplement any of that logic, it just calls it.
    try:
        result = handle_message(req.message, state["dialogue_manager"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    tier = result["tier"]
    generated = (tier == "routine")  # emergency/urgent are fixed responses; only routine calls the LLM

    # Optional: if your pipeline exposes a confidence/probability for
    # the tier decision (e.g. from the semantic classifier's own
    # self-reported confidence, or a margin derived from its logits),
    # surface it here so the GUI can render it next to the tier badge.
    # Safe to leave absent -- the GUI degrades gracefully without it.
    confidence = result.get("confidence")

    # Same opt-in pattern as confidence above. dual_layer_safety.py's
    # orchestrator already knows which sub-layer fired (that's how it
    # decides whether to short-circuit the semantic call) -- if it adds
    # keyword_triggered/semantic_triggered to the result dict, the GUI
    # will show real per-layer state instead of "—" (unknown).
    keyword_triggered = result.get("keyword_triggered")
    semantic_triggered = result.get("semantic_triggered")

    return ChatResponse(
        session_id=session_id,
        tier=tier,
        text=result["response"],
        turn=state["turn"],
        mode=result.get("mode"),
        generated=generated,
        confidence=confidence,
        keyword_triggered=keyword_triggered,
        semantic_triggered=semantic_triggered,
    )


@app.post("/reset")
def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "ok", "offline": True}


# Serve the chat frontend as static files at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
