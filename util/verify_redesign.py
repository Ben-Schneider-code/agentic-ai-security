#!/usr/bin/env python3
"""Verification harness for the self-play redesign — runs without GPU.

Tests:
  1. Module imports — env files load cleanly after refactor.
  2. Honeypot taxonomy — get_total_honeypots returns 30/34/64 by arm.
  3. Honeypot detection — column hits, row hits, legitimate-query non-hits.
  4. Min-length floor — replaces deleted degeneracy heuristics.
  5. Benign pool partition — vanilla/bordercase counts match plan, deterministic.
  6. build_benign_pool — sizes 120/20 produce 140 unique items, 0/0 produces 0.
  7. vLLM config generator — multi-LoRA pool emits all adapters.
  8. metrics.emit_per_epoch_metrics — degrades gracefully on empty input.

Usage:
    python3 util/verify_redesign.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _failures() -> list[str]:
    return _failures_list


_failures_list: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {name}{(' — ' + detail) if detail else ''}")
    if not condition:
        _failures_list.append(name)


def stub_minimal_modules() -> None:
    """Stub heavy deps (numpy, pandas, util.mcp_client, constants)."""
    for name in ("numpy", "pandas", "util", "util.mcp_client", "constants"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["util.mcp_client"].MCPClient = type("MCPClient", (), {})
    sys.modules["constants"].get_runtime_model_id = lambda: ("stub-model",)


def _load_redteam_env(honeypot_type: str | None = None):
    """Load the redteam env module fresh, optionally with a HONEYPOT_TYPE."""
    if honeypot_type is not None:
        os.environ["HONEYPOT_TYPE"] = honeypot_type
    elif "HONEYPOT_TYPE" in os.environ:
        del os.environ["HONEYPOT_TYPE"]
    # Drop cached module if present so we re-read env at import.
    sys.modules.pop("marft.envs.redteam_sql.redteam_sql_env", None)
    sys.modules.pop("rt_env_v", None)

    sys.modules.setdefault("marft", types.ModuleType("marft"))
    sys.modules.setdefault("marft.envs", types.ModuleType("marft.envs"))
    sys.modules.setdefault("marft.envs.redteam_sql", types.ModuleType("marft.envs.redteam_sql"))

    spec = importlib.util.spec_from_file_location(
        "marft.envs.redteam_sql.redteam_sql_env",
        REPO_ROOT / "MARFT/marft/envs/redteam_sql/redteam_sql_env.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["marft.envs.redteam_sql.redteam_sql_env"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_blueteam_env():
    sys.modules.pop("bt_env_v", None)
    spec = importlib.util.spec_from_file_location(
        "bt_env_v",
        REPO_ROOT / "MARFT/marft/envs/blueteam_sql/blueteam_sql_env.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_imports() -> None:
    print("== 1. Module imports ==")
    rt = _load_redteam_env(honeypot_type="rowcol")
    check("redteam_sql_env imports", rt is not None)
    bt = _load_blueteam_env()
    check("blueteam_sql_env imports", bt is not None)


def test_honeypot_taxonomy() -> None:
    print("== 2. Honeypot taxonomy ==")
    rc = _load_redteam_env(honeypot_type="rowcol")
    rr = _load_redteam_env(honeypot_type="row")
    rcol = _load_redteam_env(honeypot_type="col")

    check("rowcol total = 64", rc.get_total_honeypots() == 64,
          f"got {rc.get_total_honeypots()}")
    check("row total = 30", rr.get_total_honeypots() == 30,
          f"got {rr.get_total_honeypots()}")
    check("col total = 34", rcol.get_total_honeypots() == 34,
          f"got {rcol.get_total_honeypots()}")

    # Sanity: the col arm's columns_by_table must enumerate all 5 forbidden tables.
    cbt = rcol.get_active_honeypot_columns_by_table()
    expected = {"customer", "address", "customeraddress", "salesorderheader", "salesorderdetail"}
    check("col arm has all 5 forbidden tables", set(cbt.keys()) == expected,
          f"got {sorted(cbt.keys())}")

    # Reset to default
    _load_redteam_env(honeypot_type="rowcol")


def test_honeypot_detection() -> None:
    print("== 3. Honeypot detection ==")
    rc = _load_redteam_env(honeypot_type="rowcol")

    # Legitimate query
    hits = rc.detect_honeypot_access("29485|Self", sql_query="SELECT * FROM customer WHERE customerid=29485")
    check("legitimate customer query → 0 hits", len(hits) == 0,
          f"got {len(hits)} hits: {hits}")

    # Forbidden col
    hits = rc.detect_honeypot_access(
        "orlando0@adventure-works.com",
        sql_query="SELECT emailaddress FROM customer WHERE customerid=29485",
    )
    cols = [(h["type"], h["identifier"]) for h in hits if h["type"].startswith("column_access")]
    check("emailaddress detected on customer", ("column_access:customer", "emailaddress") in cols,
          f"hits={hits}")

    # Cross-table forbidden
    hits = rc.detect_honeypot_access("1\n2\n3", sql_query="SELECT customerid FROM customeraddress")
    types_ = [h["type"] for h in hits]
    check("customerid on customeraddress flagged", "column_access:customeraddress" in types_,
          f"types={types_}")

    # Wide-open query → row honeypots
    res = "1|Orlando|orlando0@aw.com\n2|Keith\n3|Donna\n29485|Self"
    hits = rc.detect_honeypot_access(res, sql_query="SELECT * FROM customer")
    row_ids = sorted(h["identifier"] for h in hits if h["type"] == "owner_id")
    check("wide-open SELECT * hits owner_ids 1,2,3", row_ids == [1, 2, 3],
          f"got {row_ids}")

    # row arm: column hits should be empty
    rr = _load_redteam_env(honeypot_type="row")
    hits = rr.detect_honeypot_access(
        "x", sql_query="SELECT emailaddress FROM customer WHERE customerid=29485"
    )
    col_hits = [h for h in hits if h["type"].startswith("column_access")]
    check("row arm zeroes out column detection", len(col_hits) == 0,
          f"got {col_hits}")

    # col arm: row hits should be empty
    rcol = _load_redteam_env(honeypot_type="col")
    hits = rcol.detect_honeypot_access("1\n2\n3", sql_query="SELECT * FROM customer")
    row_hits = [h for h in hits if h["type"] == "owner_id"]
    check("col arm zeroes out row detection", len(row_hits) == 0,
          f"got {row_hits}")

    _load_redteam_env(honeypot_type="rowcol")


def test_min_length_floor() -> None:
    print("== 4. Min-length floor (degeneracy replacement) ==")
    rc = _load_redteam_env(honeypot_type="rowcol")
    is_short, reason = rc.detect_short_input("hi")
    check("short input flagged", is_short and "too_short" in reason)
    is_short, _ = rc.detect_short_input("this is a longer attack input here")
    check("long input not flagged", not is_short)
    is_short, reason = rc.detect_short_input(None)
    check("None input flagged", is_short and reason == "null_input")


def test_benign_pool_partition() -> None:
    print("== 5. Benign pool partition ==")
    bt = _load_blueteam_env()
    plain = len(bt._PLAIN_POOL)
    adv = len(bt._ADVERSARIAL_POOL)
    check("plain pool >= 120", plain >= 120, f"got {plain}")
    check("adversarial pool >= 20", adv >= 20, f"got {adv}")
    # Determinism: re-import should produce the same first-N
    bt2 = _load_blueteam_env()
    check("partition deterministic across reloads",
          bt._PLAIN_POOL == bt2._PLAIN_POOL and bt._ADVERSARIAL_POOL == bt2._ADVERSARIAL_POOL)


def test_build_benign_pool() -> None:
    print("== 6. build_benign_pool ==")
    bt = _load_blueteam_env()
    pool = bt.build_benign_pool(120, 20)
    check("120/20 → 140 entries", len(pool) == 140, f"got {len(pool)}")
    check("entries all unique", len(set(pool)) == len(pool))
    pool0 = bt.build_benign_pool(0, 0)
    check("0/0 → 0 entries", len(pool0) == 0)
    pool_smaller = bt.build_benign_pool(60, 10)
    check("60/10 → 70 entries", len(pool_smaller) == 70)
    # Deterministic across calls
    p1 = bt.build_benign_pool(60, 10)
    p2 = bt.build_benign_pool(60, 10)
    check("build_benign_pool deterministic", p1 == p2)


def test_vllm_config_generator() -> None:
    print("== 7. vLLM config generator (multi-LoRA pool) ==")
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        # Stub adapters with real (empty) directories so generator validation passes
        a1 = os.path.join(tmp, "a1"); os.makedirs(a1)
        a2 = os.path.join(tmp, "a2"); os.makedirs(a2)
        a3 = os.path.join(tmp, "a3"); os.makedirs(a3)
        registry = {
            "selfplay_id": "test",
            "entries": [
                {"iter": 1, "name": "red_iter_1", "path": a1},
                {"iter": 2, "name": "red_iter_2", "path": a2},
                {"iter": 3, "name": "red_iter_3", "path": a3},
            ],
        }
        reg_path = os.path.join(tmp, "registry.json")
        with open(reg_path, "w") as f:
            json.dump(registry, f)

        out_path = os.path.join(tmp, "config.json")
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "util/generate_vllm_config.py"),
                "--target", "blueteam",
                "--opponent-lora-pool", reg_path,
                "--out-config", out_path,
                "--actor-gpu", "0",
            ],
            capture_output=True, text=True,
        )
        check("generator exits 0", result.returncode == 0, result.stderr)
        cfg = json.load(open(out_path))
        check("pool_size = 3", cfg.get("_pool_size") == 3)
        check("pool_names match", cfg.get("_pool_names") == ["red_iter_1", "red_iter_2", "red_iter_3"])
        extra = cfg["servers"][0]["extra_args"]
        n_modules = sum(
            1 for x in extra
            if isinstance(x, str) and x.startswith("red_iter_") and "=" in x
        )
        check("emits 3 --lora-modules entries", n_modules == 3, f"saw {n_modules}")


def test_metrics_emit() -> None:
    print("== 8. metrics.emit_per_epoch_metrics ==")
    # Drop the `util` stub installed earlier so the real package can resolve.
    for k in list(sys.modules):
        if k == "util" or k.startswith("util."):
            del sys.modules[k]
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "metrics_mod", REPO_ROOT / "util/metrics.py"
    )
    metrics_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_mod)
    summary = metrics_mod.emit_per_epoch_metrics(
        "/nonexistent/reward_debug.jsonl", prefix="[test]"
    )
    check("empty input returns {}", summary == {})


def main() -> int:
    stub_minimal_modules()
    print("Verifying self-play redesign (no GPU needed)...\n")
    test_imports()
    test_honeypot_taxonomy()
    test_honeypot_detection()
    test_min_length_floor()
    test_benign_pool_partition()
    test_build_benign_pool()
    test_vllm_config_generator()
    test_metrics_emit()
    print()
    if _failures():
        print(f"\033[91m{len(_failures())} check(s) failed:\033[0m " + ", ".join(_failures()))
        return 1
    print("\033[92mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
