"""
test_model.py

The very first script: confirms you can talk to a local model from
Python (not just the ollama terminal command), and times how long it
takes. Everything else in this project builds on this working.
"""

import ollama
import time

start = time.time()
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What are the classic symptoms of appendicitis?"}],
)
elapsed = time.time() - start

print(response["message"]["content"])
print(f"\n--- Took {elapsed:.2f} seconds ---")
