"""
gather_repro_info.py

Collects most of what repro_log.md needs into one place, so you don't
have to hunt down and run each command separately. Run this, then
copy the printed output into the matching sections of repro_log.md
and check off the boxes.

Some items (corpus provenance, chunking config) can't be auto-detected
and are listed as reminders at the end instead.
"""

import subprocess
import sys


def run(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        return output if output else "(no output -- command may have failed, check manually)"
    except Exception as e:
        return f"(could not run automatically: {e})"


def main():
    print("=" * 60)
    print("PASTE EVERYTHING BELOW INTO repro_log.md")
    print("=" * 60)

    print("\n--- Ollama version ---")
    print(run("ollama --version"))

    print("\n--- Models pulled (ollama list) ---")
    print(run("ollama list"))

    print("\n--- Model details (run per model you tested) ---")
    for model in ["llama3.2", "qwen2.5", "gemma2:2b"]:
        print(f"\n$ ollama show {model}")
        print(run(f"ollama show {model}"))

    print("\n--- Python version ---")
    print(sys.version)

    print("\n--- Relevant installed package versions ---")
    print(run('pip freeze | grep -iE "ollama|chromadb|sentence-transformers|datasets|psutil"'))

    print("\n--- Hardware specs (Mac) ---")
    print(run('system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Memory"'))

    print("\n" + "=" * 60)
    print("STILL NEEDS TO BE FILLED IN BY HAND (can't be auto-detected):")
    print("=" * 60)
    print("""
- Corpus provenance table: for each file in docs/, note its source
  URL (or "self-authored placeholder" for the sample files), license/
  reuse terms, and the date you added it.

- Chunking configuration: currently fixed-length, 500 characters,
  no overlap (from build_index.py). Update this note if you change
  the chunking approach later.

- GPU-tier hardware specs: not applicable yet since you're only on
  the laptop_cpu tier so far. Fill this in once you run the Colab
  GPU tier.
""")


if __name__ == "__main__":
    main()
