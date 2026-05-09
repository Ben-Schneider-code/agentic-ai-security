#!/usr/bin/env python3
"""Count tokens in the blue-team (victim) system prompt under the Llama-3.1 tokenizer.

Used to back the ``2{,}116 tokens'' figure cited in methodology.tex. Run this
when the prompt changes to keep the paper number honest.

Self-contained: parses ``sql_system_prompt`` out of ``redteam_sql_env.py`` via
``ast`` and re-exec's that single assignment (with the module-local
``user_id`` binding substituted in), so it does not need to import MARFT /
torch. Only the HuggingFace tokenizer is loaded.

Run:
    python util/count_system_prompt_tokens.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "MARFT" / "marft" / "envs" / "redteam_sql" / "redteam_sql_env.py"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def extract_prompt(source: str) -> str:
    tree = ast.parse(source)
    user_id_value = None
    prompt_node = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "user_id":
            user_id_value = ast.literal_eval(node.value)
        elif target.id == "sql_system_prompt":
            prompt_node = node
    if user_id_value is None:
        sys.exit(f"ERROR: could not find `user_id = ...` in {ENV_FILE}")
    if prompt_node is None:
        sys.exit(f"ERROR: could not find `sql_system_prompt = ...` in {ENV_FILE}")

    module = ast.Module(body=[prompt_node], type_ignores=[])
    ns: dict = {"user_id": user_id_value}
    exec(compile(module, str(ENV_FILE), "exec"), ns)
    return ns["sql_system_prompt"]


def main() -> None:
    source = ENV_FILE.read_text()
    prompt = extract_prompt(source)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"tokenizer        : {MODEL_ID}")
    print(f"character length : {len(prompt):,}")
    print(f"line count       : {prompt.count(chr(10)) + 1}")
    print(f"token count      : {len(tokens):,}  (add_special_tokens=False)")


if __name__ == "__main__":
    main()
