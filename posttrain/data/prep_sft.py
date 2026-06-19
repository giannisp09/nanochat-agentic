"""
Build a light tool-use + coding SFT dataset for Track A (format primer).

Qwen3 already does tool use, so this stage just locks in the write->run->finalize
format. We emit chat-format JSONL (messages with roles), starting from the
bundled CodingToolTrace (authentic sandbox outputs). For the real run, MIX IN
open agentic/tool-use datasets and subsample to ~20-50K, e.g.:
  - nvidia/Nemotron-SFT-Agentic-v2
  - glaiveai/glaive-function-calling-v2
  - Team-ACE/ToolACE
Use each model's NATIVE tool/chat template at tokenization time (verl/TRL apply it
from the tokenizer); do NOT hard-code nanochat's <|python_start|> tokens here.

    python -m posttrain.data.prep_sft --out data/sft_toolcode.jsonl --repeat 50
"""

import argparse
import json
import os

from tasks.tooltrace import CodingToolTrace


def _to_chat_messages(conv):
    """Flatten our multi-part assistant content into plain text messages for SFT.

    Track A serves HF models whose chat template wants string content; we render
    the tool turns inline as text. (Track B keeps the structured parts + special
    tokens.) Adjust the rendering to match your base model's tool convention.
    """
    out = []
    for m in conv["messages"]:
        if m["role"] == "user" or isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                        else "".join(p["text"] for p in m["content"])})
            continue
        # assistant with parts -> stitch into one string with fenced tool I/O
        chunks = []
        for p in m["content"]:
            if p["type"] == "text":
                chunks.append(p["text"])
            elif p["type"] == "python":
                chunks.append(f"\n```python\n{p['text']}\n```\n")
            elif p["type"] == "python_output":
                chunks.append(f"```output\n{p['text']}```\n")
        out.append({"role": "assistant", "content": "".join(chunks)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Build a tool-use+coding SFT jsonl (format primer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="oversample the small bundled set this many times")
    args = ap.parse_args()

    trace = CodingToolTrace(split="train")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for _ in range(args.repeat):
            for i in range(len(trace)):
                f.write(json.dumps({"messages": _to_chat_messages(trace[i])}) + "\n")
                n += 1
    print(f"Wrote {n} SFT conversations -> {args.out}")
    print("NOTE: mix in real agentic/tool datasets (Nemotron-SFT-Agentic-v2, ToolACE, glaive-fc-v2) for a strong primer.")


if __name__ == "__main__":
    main()
