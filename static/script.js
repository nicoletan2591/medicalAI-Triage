(() => {
  "use strict";

  const log = document.getElementById("log");
  const composer = document.getElementById("composer");
  const input = document.getElementById("msg-input");
  const sendBtn = document.getElementById("send-btn");
  const resetBtn = document.getElementById("reset-btn");
  const resetBtnMobile = document.getElementById("reset-btn-mobile");
  const acuityTrack = document.getElementById("acuity-rail").querySelector(".acuity-track");
  const acuityMarker = document.getElementById("acuity-marker");
  const offlinePill = document.getElementById("offline-pill");
  const offlinePillMobile = document.getElementById("offline-pill-mobile");
  const layerKeyword = document.getElementById("layer-keyword");
  const layerSemantic = document.getElementById("layer-semantic");
  const micBtn = document.getElementById("mic-btn");
  const langToggle = document.getElementById("lang-toggle");
  const voiceNote = document.getElementById("voice-note");

  const HEALTH_POLL_MS = 15000;

  const tplMessage = document.getElementById("tpl-message");
  const tplAgentTurn = document.getElementById("tpl-agent-turn");

  let sessionId = null;

  const TIER_LABEL = {
    routine: "Routine",
    urgent: "Urgent",
    emergency: "Emergency",
  };

  // ---- helpers ---------------------------------------------------

  function scrollToBottom() {
    log.scrollTop = log.scrollHeight;
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  }

  function setAcuityState(tier) {
    acuityTrack.classList.remove("state-routine", "state-urgent", "state-emergency");
    acuityMarker.classList.remove("state-routine", "state-urgent", "state-emergency");
    if (tier) {
      acuityTrack.classList.add(`state-${tier}`);
      acuityMarker.classList.add(`state-${tier}`);
    }
  }

  // Layer indicators default to "unknown" (—) rather than a false
  // "clear" -- they only show a real state once the backend actually
  // reports keyword_triggered / semantic_triggered for a turn.
  function setLayerState(el, value) {
    const stateEl = el.querySelector(".layer-state");
    if (value === true) {
      el.dataset.state = "triggered";
      stateEl.textContent = "TRIGGERED";
    } else if (value === false) {
      el.dataset.state = "clear";
      stateEl.textContent = "CLEAR";
    } else {
      el.dataset.state = "unknown";
      stateEl.textContent = "—";
    }
  }

  function setLayersChecking() {
    [layerKeyword, layerSemantic].forEach((el) => {
      el.dataset.state = "checking";
      el.querySelector(".layer-state").textContent = "…";
    });
  }

  // ---- health polling ----------------------------------------------

  async function checkHealth() {
    [offlinePill, offlinePillMobile].forEach((el) => { el.dataset.health = "checking"; });
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("bad status");
      await res.json();
      [offlinePill, offlinePillMobile].forEach((el) => {
        el.dataset.health = "ok";
        el.title = "Backend reachable on this device";
      });
    } catch (e) {
      [offlinePill, offlinePillMobile].forEach((el) => {
        el.dataset.health = "unreachable";
        el.title = "Backend not responding — check the server is running";
      });
    }
  }

  function removeIntro() {
    const intro = document.getElementById("intro");
    if (intro) intro.remove();
  }

  function addUserMessage(text) {
    removeIntro();
    const node = tplMessage.content.cloneNode(true);
    node.querySelector(".msg-bubble").textContent = text;
    log.appendChild(node);
    scrollToBottom();
  }

  function addThinkingIndicator() {
    const wrap = document.createElement("div");
    wrap.className = "agent-turn thinking";
    wrap.innerHTML = `
      <div class="msg-bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>`;
    log.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addAgentTurn(result) {
    const { tier, text, mode, generated, confidence, keyword_triggered, semantic_triggered } = result;

    setLayerState(layerKeyword, keyword_triggered);
    setLayerState(layerSemantic, semantic_triggered);

    const node = tplAgentTurn.content.cloneNode(true);
    const wrapper = node.querySelector(".agent-turn");
    const badge = node.querySelector(".turn-badge");
    const bubble = node.querySelector(".msg-bubble");
    const meta = node.querySelector(".turn-meta");

    wrapper.classList.add(`tier-${tier}`);
    if (!generated) {
      // emergency/urgent fixed responses — the escalation path
      wrapper.classList.add("escalated");
    }

    badge.classList.add(`tier-${tier}`);
    let badgeHtml = TIER_LABEL[tier] || tier;
    if (typeof confidence === "number") {
      badgeHtml += ` <span class="conf">· ${Math.round(confidence * 100)}%</span>`;
    }
    badge.innerHTML = badgeHtml;

    bubble.textContent = text;

    // meta line: clarifying-question / assessment state for routine
    // turns, turn count otherwise.
    if (tier === "routine" && mode) {
      const pillLabel = mode === "clarify" ? "Gathering more information" : "Assessment";
      const modePillEl = document.createElement("span");
      modePillEl.className = `mode-pill mode-${mode}`;
      modePillEl.textContent = pillLabel;
      meta.appendChild(modePillEl);
    } else if (!generated) {
      meta.textContent = "Fixed response — not model-generated";
    }

    log.appendChild(node);
    setAcuityState(tier);
    scrollToBottom();
  }

  function addErrorNote(message) {
    const el = document.createElement("div");
    el.className = "error-note";
    el.textContent = message;
    log.appendChild(el);
    scrollToBottom();
  }

  // ---- voice input ---------------------------------------------------
  //
  // Uses the browser's SpeechRecognition API. Important: this is NOT
  // on-device recognition -- Chrome (and most implementations) send
  // the audio to a cloud speech service to be transcribed. Typed
  // input in this app never leaves the device; voice input does.
  // Transcribed text always lands in the textarea for the person to
  // review/edit -- it is never auto-submitted -- partly so a
  // mis-transcription can't silently become what gets sent.

  let recognitionLang = "en-US";
  let recognition = null;
  let listening = false;

  function showVoiceNote(text, isError) {
    voiceNote.textContent = text;
    voiceNote.hidden = !text;
    voiceNote.classList.toggle("is-error", !!isError);
  }

  const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognitionCtor) {
    micBtn.disabled = true;
    micBtn.title = "Voice input isn't supported in this browser";
  } else {
    recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = recognitionLang;

    let baseText = ""; // textarea content at the moment recording started

    recognition.addEventListener("start", () => {
      listening = true;
      micBtn.dataset.listening = "true";
      langToggle.dataset.disabled = "true";
      baseText = input.value ? input.value.trim() + " " : "";
      const label = recognitionLang === "zh-CN" ? "正在聆听…" : "Listening…";
      showVoiceNote(label, false);
    });

    recognition.addEventListener("result", (e) => {
      let transcript = "";
      for (let i = 0; i < e.results.length; i++) {
        transcript += e.results[i][0].transcript;
      }
      input.value = baseText + transcript;
      autoGrow();
    });

    recognition.addEventListener("error", (e) => {
      const messages = {
        "not-allowed": recognitionLang === "zh-CN" ? "麦克风访问被拒绝" : "Microphone access denied",
        "no-speech": recognitionLang === "zh-CN" ? "未检测到语音" : "No speech detected",
        "network": recognitionLang === "zh-CN" ? "语音服务不可用（需要网络连接）" : "Speech service unavailable (needs network access)",
        "audio-capture": recognitionLang === "zh-CN" ? "未检测到麦克风" : "No microphone detected — check it's connected and not in use by another app",
      };
      showVoiceNote(messages[e.error] || `Voice input error: ${e.error}`, true);
    });

    recognition.addEventListener("end", () => {
      listening = false;
      micBtn.dataset.listening = "false";
      langToggle.dataset.disabled = "false";
      if (!voiceNote.classList.contains("is-error")) showVoiceNote("", false);
      input.focus();
    });

    micBtn.addEventListener("click", () => {
      if (listening) {
        recognition.stop();
        return;
      }
      recognition.lang = recognitionLang;
      try {
        recognition.start();
      } catch (e) {
        // start() throws if called while already active in some
        // browsers -- surfacing rather than silently failing.
        showVoiceNote("Couldn't start voice input — try again.", true);
      }
    });

    langToggle.querySelectorAll(".lang-opt").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (listening) return; // language is locked mid-recording
        recognitionLang = btn.dataset.lang;
        langToggle.querySelectorAll(".lang-opt").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
    });
  }

  // ---- networking --------------------------------------------------

  async function sendMessage(message) {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  async function resetSession() {
    if (sessionId) {
      try {
        await fetch(`/reset?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
      } catch (_) {
        /* best-effort */
      }
    }
    sessionId = null;
    log.innerHTML = `
      <div class="intro" id="intro">
        <h1 class="intro-title">What's going on?</h1>
        <p class="intro-sub">This runs entirely on this device — nothing leaves it. It maps symptoms to a triage category and does not diagnose or replace clinical judgment.</p>
      </div>`;
    setAcuityState(null);
    setLayerState(layerKeyword, null);
    setLayerState(layerSemantic, null);
    input.value = "";
    autoGrow();
    input.focus();
  }

  // ---- events -----------------------------------------------------

  input.addEventListener("input", autoGrow);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });

  composer.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addUserMessage(text);
    input.value = "";
    autoGrow();
    input.disabled = true;
    sendBtn.disabled = true;

    const thinkingEl = addThinkingIndicator();
    setLayersChecking();

    try {
      const result = await sendMessage(text);
      sessionId = result.session_id;
      thinkingEl.remove();
      addAgentTurn(result);
    } catch (err) {
      thinkingEl.remove();
      addErrorNote(err.message || "Something went wrong reaching the local agent.");
      setLayerState(layerKeyword, null);
      setLayerState(layerSemantic, null);
      checkHealth(); // a failed /chat call is a good moment to re-verify reachability
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  });

  resetBtn.addEventListener("click", () => {
    resetSession();
  });

  resetBtnMobile.addEventListener("click", () => {
    resetSession();
  });

  autoGrow();
  checkHealth();
  setInterval(checkHealth, HEALTH_POLL_MS);
})();
