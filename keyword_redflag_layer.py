"""
keyword_redflag_layer.py

Layer (a) of the dual-layer safety system: deterministic keyword/rule
matching for clearly-defined red-flag terms and presentations.

This is intentionally simple and auditable -- plain string/regex
matching, no model involved -- so the safety-critical path here does
not depend on generative model behavior, per the design principle in
the methodology. It will have false negatives on paraphrased or
implicit red flags (that's what the semantic layer in
semantic_intent_classifier.py is for), but it should have very close
to zero false negatives on cases stated in plain, direct language.

The term list is sourced directly from redflag_symptoms_qa_50.txt and
specific_presentations_qa_50.txt in the corpus, so this layer stays
consistent with the rest of the project rather than being a separately
invented list.

CHANGE FROM PRIOR (BINARY) VERSION -- READ THIS:
Your original pattern list was authored entirely for a binary
escalate/don't-escalate decision, which means nearly every pattern in
it already represents an EMERGENCY-tier concern by construction --
there was no "urgent, not quite emergency" bucket for a keyword to
belong to. Splitting the list therefore mostly means: everything stays
in EMERGENCY_PATTERNS, with exactly one deliberate exception pulled
into URGENT_PATTERNS below (chest tightness -- see that section for
why). This is not a case of me evenly dividing the list; it's a
reflection of what the list actually is.

Practical implication: right now, "urgent" is reached almost entirely
through the semantic layer, not this one. If you want the keyword
layer itself to catch more genuinely-urgent (not emergency) cases --
e.g. "worsening," "won't go away," "getting worse" style language --
that requires AUTHORING NEW PATTERNS, not just re-tiering old ones.
That's a real to-do, not something this upgrade did for you; flagged
here so it doesn't get silently assumed as done.

Known accepted limitation (unchanged from prior version): some
patterns (e.g. "chest pain") will false-positive on presentations that
turn out to be benign. This was found during comprehensive testing
(test_keyword_layer_comprehensive.py) and left as-is rather than
patched with exclusion words like "mild," since suppressing on such
words risks creating a false negative for a genuinely dangerous case
that happens to also start with "mild" before escalating.
Over-triggering toward caution remains the accepted trade-off,
consistent with the corpus's own stated triage principle of erring
toward the higher-acuity interpretation when uncertain. The one
partial exception is documented below.
"""

import re

# ---------------------------------------------------------------
# EMERGENCY_PATTERNS -- immediate, potentially life-threatening.
# This is your original RED_FLAG_PATTERNS list, unchanged, MINUS the
# "chest tightness" case split out below. Everything else here was
# already written to indicate an emergency-level concern in the
# original binary system, so it stays at the same tier.
# ---------------------------------------------------------------
EMERGENCY_PATTERNS = [
    # Cardiac / circulatory
    r"\bchest (pain|pressure)\b",  # split from original "(pain|pressure|tightness)" --
                                    # see URGENT_PATTERNS below for why tightness moved
    r"\bpain (radiating|spreading) to (my |the )?(arm|jaw|back)\b",
    r"\b(fainted|fainting|passed out|lost consciousness)\b",
    r"\bshortness of breath\b.*\b(rest|sitting|sudden)\b",
    r"\bsudden severe (shortness of breath|breathlessness)\b",
    r"\b(irregular heartbeat|heart.{0,10}beating irregularly|heart.{0,10}irregular)\b.*\b(dizzy|dizziness|chest)\b",

    # Neurological / stroke
    r"\b(face|speech) (droop|drooping|slurred|slurring)\b",
    r"\b(face|speech).{0,20}\b(droop|slur)",
    r"\bsudden (weakness|numbness)\b",
    r"\b(can't|cannot|can not|difficulty) (speak|speaking|talking)\b",
    r"\bworst headache of my life\b",
    r"\bsudden (severe|worst) headache\b",
    r"\bfirst.time seizure\b",
    r"\bseizure\b.*\b(minutes|long|won't stop|wont stop)\b",
    r"\bloss of consciousness\b.*\bhead\b",

    # Respiratory
    r"\b(can't|cannot|can not) breathe\b",
    r"\bgasping for air\b",
    r"\bsevere.{0,10}(breathing|breath)\b",
    r"\b(lips?|face|skin).{0,15}(turn(ed|ing)? blue|blue.{0,5}(tint|color|colour))\b",
    r"\bblue (lips|face|skin)\b",
    r"\bchoking\b.*\b(can't|cannot|unable)\b",

    # Abdominal / GI
    r"\bsevere.{0,15}abdominal pain\b",  # NOTE: broad pattern, kept at emergency per
                                          # err-high default -- worth your own review,
                                          # since "severe abdominal pain" alone (no
                                          # vomiting/rigidity/bleeding) is a case some
                                          # protocols would place at urgent instead.
    r"\bvomiting blood\b",
    r"\bblack.{0,10}(tarry )?stool",
    r"\bstool.{0,15}(black|tarry)",
    r"\brigid.{0,10}abdomen\b",
    r"\babdomen.{0,15}rigid\b",

    # Mental health crisis
    r"\b(intend|intent|plan|going|want) to (harm|hurt|kill) (myself|himself|herself|themselves|someone|others)\b",
    r"\bsuicidal\b",
    r"\bkill myself\b",
    r"\bend my life\b",

    # Allergic / anaphylaxis
    r"\b(facial|throat|tongue|lip) swelling\b.*\b(breath|breathing)\b",
    r"\bthroat.{0,10}(tight|tightness|closing)\b",
    r"\bsevere allergic reaction\b",

    # Bleeding / shock
    r"\buncontrolled bleeding\b",
    r"\bbleeding.{0,15}(won't stop|wont stop|not stopping)\b",
    r"\b(cold|clammy) skin\b.*\b(pale|weak pulse)\b",

    # Infection / sepsis
    r"\bhigh fever\b.*\b(confusion|confused)\b",
    r"\brash.{0,15}(doesn't fade|does not fade|won't fade)\b",

    # Pediatric-specific
    r"\binfant\b.*\bfever\b",
    r"\b(unusually|extremely) drowsy\b.*\bwake\b",

    # Pregnancy-specific
    r"\bpregnant\b.*\b(bleeding|severe headache|vision)\b",
]

# ---------------------------------------------------------------
# URGENT_PATTERNS -- concerning, needs care soon, not an immediate
# life threat. Deliberately thin right now -- see module docstring.
# ---------------------------------------------------------------
URGENT_PATTERNS = [
    # Split out of the original combined chest pattern. Your own
    # docstring already documented "mild chest tightness after a big
    # meal" as a known false-positive on the binary system -- pain and
    # pressure stay at emergency (classic ACS descriptors), but
    # tightness alone (without pain/pressure) is a softer signal and
    # now has a real place to land instead of over-triggering to
    # emergency or getting silently suppressed.
    r"\bchest tightness\b",

    # Knocked-out tooth is genuinely time-sensitive (best outcome if
    # reimplanted within ~30-60 min) but is not life-threatening, so
    # by the tier definitions (emergency = immediate life threat) it
    # belongs here, not in EMERGENCY_PATTERNS. Flagging the tension
    # explicitly: "urgent" normally means hours, this specifically
    # means minutes. If that distinction matters for your triage
    # response wording, consider a dedicated fixed response for this
    # one case rather than the generic URGENT response.
    r"\bknocked.out\b.*\btooth\b",
]

_COMPILED_EMERGENCY = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]
_COMPILED_URGENT = [re.compile(p, re.IGNORECASE) for p in URGENT_PATTERNS]


def check_keyword_tier(message):
    """Checks a message against the two-tier red-flag pattern lists.

    Returns (tier, matched_patterns):
        tier: 'emergency', 'urgent', or None (no pattern matched)
        matched_patterns: list[str] of the regex patterns that matched
                           (from whichever tier matched -- if a message
                           somehow matches both, only emergency-tier
                           matches are returned, since that's the tier
                           that determines the result)

    Intentionally simple and auditable -- no model inference, no
    ambiguity in how a match is determined.
    """
    emergency_matches = [p.pattern for p in _COMPILED_EMERGENCY if p.search(message)]
    if emergency_matches:
        return "emergency", emergency_matches

    urgent_matches = [p.pattern for p in _COMPILED_URGENT if p.search(message)]
    if urgent_matches:
        return "urgent", urgent_matches

    return None, []


def check_keyword_redflags(message):
    """Backward-compatible binary wrapper, for any other code still
    calling the old interface. Prefer check_keyword_tier() in new code
    -- this collapses urgent and emergency into a single True, exactly
    the granularity the three-tier upgrade exists to add back.
    """
    tier, matches = check_keyword_tier(message)
    return tier is not None, matches
