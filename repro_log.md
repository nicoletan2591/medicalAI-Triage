# Reproducibility Log

- [x] **Quantization commands used per model**
  - `ollama pull llama3.2` → tag: `llama3.2:latest` (2.0 GB) — architecture: llama, parameters: 3.2B, quantization: **Q4_K_M**, context length: 131072, license: Llama 3.2 Community License Agreement
  - `ollama pull qwen2.5` → tag: `qwen2.5:latest` (4.7 GB) — architecture: qwen2, parameters: 7.6B, quantization: **Q4_K_M**, context length: 32768, license: Apache License 2.0
  - `ollama pull gemma2:2b` → tag: `gemma2:2b` (1.6 GB) — architecture: gemma2, parameters: 2.6B, quantization: **Q4_0**, context length: 8192, license: Gemma Terms of Use
  - Note: default `gemma2:latest` (5.4 GB, 9B parameters) was also pulled but is **not used for benchmarking** — it caused a full system freeze on this hardware tier (8GB unified memory) and was excluded. See Troubleshooting Notes / Phase I report for details.

- [x] **Runtime + library versions**
  - Ollama: `0.32.5`
  - Python: `3.12.2` (conda-forge build) — **NOTE: this does not match the venv's Python 3.10.0 seen in earlier sessions. This environment inconsistency is unresolved and flagged for cleanup before Phase II** (see open issue at bottom of this file).
  - Installed packages detected in this environment: `psutil==7.2.1` only. `chromadb`, `sentence-transformers`, `datasets`, and `ollama` (the Python package) were NOT found in this environment's `pip freeze`, despite being used successfully in earlier sessions — indicating those were installed in a *different* Python environment (likely the actual venv, or another conda env). This mismatch needs to be resolved by running all project scripts from a single, consistent environment going forward.

- [x] **Hardware specs per tier**
  - laptop_cpu tier: MacBook Air, Apple M3 chip, 8 GB unified memory (~5.3 GB available to model runtime per Ollama's own reported startup log).
  - GPU tier: not applicable — project scope reduced to a single hardware tier (laptop_cpu only) due to no NVIDIA GPU being available locally. Documented as a deliberate scope decision, not an oversight.

- [x] **Corpus provenance table**

  | File | Source | License / reuse terms | Date added |
  |---|---|---|---|
  | redflag_symptoms.txt | Self-authored placeholder content (not from a specific external source) | N/A — original text written for pipeline testing only | Phase I setup |
  | triage_acuity_categories.txt | Self-authored placeholder content, loosely modeled on general public triage-scale concepts | N/A — original text written for pipeline testing only | Phase I setup |
  | symptom_mapping_notes.txt | Self-authored placeholder content | N/A — original text written for pipeline testing only | Phase I setup |

  **Action needed before Phase III:** these are placeholder documents used to validate the pipeline mechanics only. Real, licensed, clinically-sourced guideline documents (per the Phase I methodology, Section 4.1) still need to be sourced and substituted before any real evaluation results are collected.

- [x] **Chunking configuration**
  - Method: fixed-length character chunking (500 characters per chunk, no overlap), as implemented in `build_index.py`.
  - Note: the Phase I methodology specifies section-boundary chunking as the intended approach, since literature suggests fixed-length chunking can weaken RAG grounding in clinical text. Current implementation is a simplified placeholder for pipeline validation; switching to section-boundary chunking is a known follow-up task before Phase III.

- [x] **Embedding model**
  - `all-MiniLM-L6-v2` via `sentence-transformers`. (Exact package version not yet confirmed in the correct environment — see environment mismatch note above; re-run `pip freeze` from the actual working environment to confirm.)

- [x] **Vector store type and version**
  - Chroma (`chromadb`), persistent client, local path `./chroma_db`. (Exact package version not yet confirmed — same environment mismatch issue.)

- [x] **Final check**
  - Reviewed. Open items for a true independent rebuild: (1) resolve the Python environment inconsistency noted above so there is one documented environment, not two; (2) confirm `chromadb` and `sentence-transformers` exact versions from that resolved environment; (3) replace placeholder corpus documents with real licensed sources before Phase III.

---

## Open issue to resolve before Phase II

Two different Python environments have been used across this project's sessions:
- Terminal sessions showing `Python 3.10.0` from `/Library/Frameworks/Python.framework` with an activated `(venv)`.
- This session's terminal showing `Python 3.12.2` from `conda-forge`, with almost none of the project's packages present.

This needs to be resolved into a single, consistent, documented environment (most likely by consistently activating the intended `venv` and confirming with `pip freeze` that all required packages — `ollama`, `chromadb`, `sentence-transformers`, `datasets`, `psutil` — are present) before Phase I can be considered fully reproducible by a second party.
