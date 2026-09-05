# Full Pipeline Runbook — Chronological Order

Run everything from ONE folder only (pick 
phase 6_core, 

## Prerequisites (once, not repeated per run)

Files that must already be in this folder:
- triage_agent.py
- dual_layer_safety.py
- semantic_intent_classifier.py
- keyword_redflag_layer.py
- prompt_templates.py (your own original file — not something I built)
- chroma_db/ (your own original Phase I index)
- phase3_agent_progress.json (your 210 real vignette texts)
- triage_relabeled_draft.csv (the AI-drafted + your-reviewed ground truth)
- run_eval.py, triage_metrics.py
- merge_predictions.py, rerun_testbed.py, diagnose_errors.py
- build_priority_review.py
- app.py + static/ folder (index.html, style.css, app.js) — only needed for the GUI

```bash
pip install -r requirements.txt   # fastapi, uvicorn, pandas, numpy, scikit-learn, scipy, matplotlib, ollama
```

Ollama must be running (in its own terminal tab, left open):
```bash
ollama serve
```

---

## Step 1 — Confirm true_tier exists in your ground-truth file

```bash
python3 -c "import pandas as pd; print(pd.read_csv('triage_relabeled_draft.csv').columns.tolist())"
```
If you see `draft_true_tier` instead of `true_tier`, rename it once you're
satisfied with the reviewed labels:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('triage_relabeled_draft.csv')
df.rename(columns={'draft_true_tier':'true_tier'}).to_csv('triage_relabeled_draft.csv', index=False)
"
```

## Step 2 — Re-run all 210 vignettes through the real pipeline

```bash
python3 benchmark.py
python3 tune_rag_params.py
python3 test_semantic_classifier.py 
python3 test_keyword_layer_comprehensive.py 
python3 test_phase2_demo.py
python3 phase3_evaluation_runner.py
python3 rerun_testbed.py
```

## Step 3 — Merge predictions into your ground-truth file

```bash
python3 merge_predictions.py
```

## Step 4 — Run the evaluation

```bash
python3 run_eval.py synthetic_ceiling_results.csv --true-col true_tier --pred-col pre_clarification_tier --out-prefix ceiling_before
python3 run_eval.py synthetic_ceiling_results.csv --true-col true_tier --pred-col final_tier --out-prefix ceiling_after

Prints the confusion matrix, per-class precision/recall/F1, Quadratic
Weighted Kappa, and Missed-Emergency Rate. Also saves
`eval_report_confusion.png` and several CSV/JSON summaries.

## Step 5 — Diagnose which layer is causing errors

```bash
python3 diagnose_errors.py
```
Breaks every misclassification down by keyword-layer vs semantic-layer
attribution. Saves `error_diagnosis.csv` with full detail.

## Step 6 — Rebuild the priority-review list (only if ground truth changed)

```bash
python3 build_priority_review.py
```
Only needed again if you've re-drafted or edited `true_tier` labels
since the last time you ran this.

## Step 7 — Launch the chat GUI (optional, separate from evaluation)

```bash
uvicorn app:app --reload
```
Then open `http://127.0.0.1:8000` in your browser — as a separate,
manual action, not part of the terminal command.

---

## Quick reference: what to run after ANY code change

Anytime you edit `keyword_redflag_layer.py`, `semantic_intent_classifier.py`,
`dual_layer_safety.py`, or `triage_agent.py`, re-run steps 2 through 5
in order — skipping straight to run_eval.py on old prediction files is
the single most common mistake in this whole process (it silently
re-scores stale data instead of your actual change).

## Before trusting ANY result, verify the file you think you changed
actually changed:

```bash
python3 -c "import dual_layer_safety as d; print(hasattr(d, 'is_borderline_urgent'))"
python3 -c "import keyword_redflag_layer as k; print(hasattr(k, 'EMERGENCY_PATTERNS'))"
python3 -c "import semantic_intent_classifier as s; print(hasattr(s, 'GENERATION_OPTIONS'))"
grep -c awaiting_clarification rerun_testbed.py
```

