# Ablation Variables

This document defines the three independent variables exposed by the self-play
redesign and proves their isolation. The orchestrator (`run_ablation_grid.sh`)
sweeps these one-at-a-time around a baseline cell to attribute effects.

## Baseline cell

| Variable          | Baseline value | Levels swept                |
|-------------------|----------------|-----------------------------|
| `vanilla-size`    | `120`          | `120`, `60`, `0`            |
| `bordercase-size` | `20`           | `20`, `10`, `0`             |
| `honeypot-type`   | `rowcol`       | `rowcol`, `row`, `col`      |

Baseline = `(120, 20, rowcol)`. Axis-mode grid yields **7 cells**:

```
cell_0: v=120  b=20  h=rowcol   ← baseline
cell_1: v= 60  b=20  h=rowcol
cell_2: v=  0  b=20  h=rowcol
cell_3: v=120  b=10  h=rowcol
cell_4: v=120  b= 0  h=rowcol
cell_5: v=120  b=20  h=row
cell_6: v=120  b=20  h=col
```

## Variable 1: `vanilla-size` — plain benign training pool size

**Levels**: `120`, `60`, `0`.

**Pool source**: `_PLAIN_POOL` in `MARFT/marft/envs/blueteam_sql/blueteam_sql_env.py:651`
— deterministic sorted partition of `BENIGN_QUERIES`. Current size = **123** (so 120 is the safe ceiling).

**Selection**: `_PLAIN_POOL[:vanilla_size]` in `build_benign_pool`
(`blueteam_sql_env.py:668-685`). Deterministic across reloads; first-N truncation.

**Pass-through**:
```
run_ablation_grid.sh  --vanilla-size V
└─ run_selfplay.sh    --vanilla-size V                  (line 253)
   └─ run_training.sh --vanilla-size V                  (line 253)
      └─ train_sql.py --vanilla-size V                  (line 164)
         └─ os.environ["VANILLA_BENIGN_SIZE"] = str(V)  (line 204)
            └─ BlueTeamSQLEnv.__init__ reads env var    (line 766)
```

**Affects**:
- Blue team training benign-query pool size (and therefore FN-penalty calibration).

**Does NOT affect**:
- Red team env: zero references to `vanilla` / `benign` / `VANILLA_BENIGN_SIZE`.
- Honeypot scoring: independent module-level constants.
- Eval pool: `mode=="test"` branch uses fixed `BENIGN_EVAL_QUERIES`
  (`blueteam_sql_env.py:761-762`) — cross-cell comparability invariant.

## Variable 2: `bordercase-size` — adversarial-style benign training pool size

**Levels**: `20`, `10`, `0`.

**Pool source**: `_ADVERSARIAL_POOL` in `blueteam_sql_env.py:655` — sorted partition of
`BENIGN_QUERIES` whose text matches `_ADVERSARIAL_MARKERS` (lines 622-636). Current size = **31** (so 20 is well below the ceiling).

**Selection**: `_ADVERSARIAL_POOL[:bordercase_size]` in `build_benign_pool`
(`blueteam_sql_env.py:685`).

**Pass-through**:
```
run_ablation_grid.sh  --bordercase-size B
└─ run_selfplay.sh    --bordercase-size B               (line 254)
   └─ run_training.sh --bordercase-size B               (line 254)
      └─ train_sql.py --bordercase-size B               (line 170)
         └─ os.environ["BORDERCASE_BENIGN_SIZE"] = str(B)  (line 206)
            └─ BlueTeamSQLEnv.__init__ reads env var    (line 768)
```

**Affects**:
- Blue team training benign-style-gap calibration (queries that read benign but
  carry adversarial framing — authority impersonation, audit/compliance, etc.).

**Does NOT affect**:
- Red team env (zero references).
- Honeypot scoring.
- Vanilla pool — independent slice.
- Eval pool — fixed.

## Variable 3: `honeypot-type` — scored honeypot universe

**Levels**: `rowcol` (full), `row` (owner-id row hits only), `col` (forbidden-column hits only).

**Pool source**: `_FULL_HONEYPOT_COLUMNS_BY_TABLE` (35 col entries × 5 tables = **34 active cols**)
and `_FULL_HONEYPOT_OWNER_IDS` (**30 row entries**) in
`MARFT/marft/envs/redteam_sql/redteam_sql_env.py:145-172`.

**Selection**: `get_active_honeypot_columns_by_table()` and `get_active_honeypot_owner_ids()`
filter the universe at scoring time
(`redteam_sql_env.py:175-186`):

| Arm     | active cols | active rows | `get_total_honeypots()` |
|---------|-------------|-------------|-------------------------|
| rowcol  | 34          | 30          | **64**                  |
| row     | 0           | 30          | **30**                  |
| col     | 34          | 0           | **34**                  |

**Pass-through**:
```
run_ablation_grid.sh  --honeypot-type H
└─ run_selfplay.sh    --honeypot-type H                 (line 255)
   └─ run_training.sh --honeypot-type H                 (line 255)
      └─ train_sql.py --honeypot-type H                 (line 176)
         └─ os.environ["HONEYPOT_TYPE"] = H             (line 202)
            └─ redteam_sql_env reads at module import   (line 136)
               (re-read on every fresh Python process; cells run in separate procs)
```

**Affects**:
- `detect_honeypot_access` (`redteam_sql_env.py:738`) — uses `get_active_*`, so
  `row` zeros all column hits and `col` zeros all row hits.
- Red team reward scoring (which honeypots count for `reward_honeypot_access`).
- Blue team FP penalty (`reward_honeypot_access = -5.0`) — same detection pipeline.
- Blue team system prompt — built from `get_active_*` so blue is only warned
  about scoreable honeypots (`redteam_sql_env.py:1276,1282`).

**Does NOT affect**:
- Benign pool composition (vanilla/bordercase slices are independent).
- Eval honeypot scoring is consistent within a cell (filter applied uniformly
  to train and test; cross-cell comparisons must respect arm difference).
- Red team prompt strategies (social-engineering categories are honeypot-agnostic).

**Caveat (single-process re-import)**: `_HONEYPOT_TYPE` is read once at module
import time. Within a single Python process you cannot switch arms by re-setting
the env var. The grid orchestrator avoids this by launching each cell as a
separate `run_selfplay.sh` invocation (separate Python process).

## Isolation guarantees

The verification harness `util/verify_redesign.py` checks every isolation
property above without needing a GPU. All 8 test groups must pass before
dispatching the grid:

1. Module imports clean after refactor.
2. Honeypot taxonomy returns `64/30/34` for `rowcol/row/col`.
3. Detection honors the arm: `row` zeros col hits; `col` zeros row hits.
4. Min-length floor replaces deleted degeneracy logic.
5. Benign pool partition: `_PLAIN_POOL ≥ 120`, `_ADVERSARIAL_POOL ≥ 20`,
   deterministic across reloads.
6. `build_benign_pool(120, 20)` → 140 entries, `(0, 0)` → 0, `(60, 10)` → 70,
   deterministic.
7. vLLM multi-LoRA pool generator emits all adapter entries.
8. `metrics.emit_per_epoch_metrics` degrades gracefully on empty input.

## Preview / smoke commands

Run these in order before dispatching the full grid:

```bash
# 1. No-GPU isolation check (~5s)
python3 util/verify_redesign.py

# 2. Dry-run the axis-mode plan — prints 7 cells with their exact commands
./run_ablation_grid.sh --mode axis --profile smoke --dry-run --gpu-pairs '0,1'

# 3. Single-cell smoke on the baseline only (~30 min on 2 GPUs, ≤400 env-steps)
./run_selfplay.sh \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --num-iterations 2 --num-env-steps 400 --horizon 5 \
    --vanilla-size 120 --bordercase-size 20 --honeypot-type rowcol \
    --redteam-gpu 0 --blueteam-gpu 1
```

After smokes pass, dispatch the full axis sweep with the `default` profile.
