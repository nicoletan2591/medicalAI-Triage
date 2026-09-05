"""
prompt_templates.py

Constrains the generative layer to symptom-mapping and triage-category
assignment, not diagnosis or treatment advice -- per the methodology's
requirement that prompt templates enforce this scope explicitly rather
than relying on the model to infer appropriate boundaries on its own.

Only reached if dual_layer_safety.safety_check() did NOT escalate --
this is the non-emergency generative path.
"""

TRIAGE_SYSTEM_PROMPT = """You are a symptom-mapping and triage-category assistant. Your ONLY job is to:
1. Gather relevant details about the person's symptom(s) through clarifying questions
2. Assign a triage acuity category (Level 1-5, per the acuity framework) once enough information is gathered

You must NEVER:
- Provide a diagnosis or name a specific medical condition as the cause
- Recommend a specific treatment, medication, or dosage
- Give open-ended medical advice beyond triage categorization

If at any point the person describes something that sounds like a medical emergency, say so plainly and recommend they seek emergency care immediately -- do not continue routine questioning.

Reference material retrieved from the triage corpus (use this to inform your questions and category assignment, but do not simply repeat it verbatim):
{retrieved_context}

Conversation so far:
{conversation_history}
"""

CLARIFYING_QUESTION_PROMPT = """Based on the conversation so far and the reference material, ask ONE clarifying question to help categorize this symptom. Follow the symptom-mapping approach: ask about onset, severity, associated symptoms, or relevant history -- whichever is most useful to ask next given what's already been established. Ask only one question. Do not diagnose or suggest treatment.

Reference material:
{retrieved_context}

Conversation so far:
{conversation_history}
"""

CATEGORY_ASSIGNMENT_PROMPT = """Based on the full conversation, assign a triage acuity category (Level 1: Immediate/Resuscitation, Level 2: Emergency, Level 3: Urgent, Level 4: Less Urgent, Level 5: Non-Urgent) and briefly explain why, referencing the specific symptoms discussed. Do NOT provide a diagnosis or treatment recommendation -- category and brief reasoning only. If any uncertainty exists, err toward the higher-acuity category.

Reference material:
{retrieved_context}

Full conversation:
{conversation_history}
"""


def build_triage_prompt(retrieved_context, conversation_history, mode="clarify"):
    """Builds the appropriate constrained prompt for the current dialogue
    stage. mode: 'clarify' to ask another question, 'assign' to give a
    final triage category."""
    if mode == "clarify":
        template = CLARIFYING_QUESTION_PROMPT
    elif mode == "assign":
        template = CATEGORY_ASSIGNMENT_PROMPT
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'clarify' or 'assign'.")

    context_text = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "(no relevant material retrieved)"
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation_history) if conversation_history else "(no prior conversation)"

    return template.format(retrieved_context=context_text, conversation_history=history_text)
