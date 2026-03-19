#!/bin/bash
set -euo pipefail

# ============================================================
# Parameter Golf — Automated H100 Run
# V=8192 SwiGLU 7L + sliding window eval
# Budget: ~30 min total. Every minute counts.
# ============================================================

START_TIME=$(date +%s)
log() { echo "[$(( $(date +%s) - START_TIME ))s] $1"; }

cd /workspace

# --- STEP 1: Clone repo ---
log "Cloning repository..."
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/alepot55/parameter-golf.git
fi
cd parameter-golf
log "Clone complete."

# --- STEP 2: Install deps ---
pip install sentencepiece 2>/dev/null || true

# --- STEP 3: Download sp1024 dataset ---
log "Downloading sp1024 dataset (80 train shards + val)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
log "Download complete."

TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
log "Found $TRAIN_SHARDS train shards, $VAL_SHARDS val shards"

if [ "$TRAIN_SHARDS" -lt 80 ] || [ "$VAL_SHARDS" -lt 1 ]; then
    log "ERROR: Missing shards! Expected 80 train + 1+ val"
    exit 1
fi

# --- STEP 4: Retokenize sp1024 → sp8192 (parallel) ---
log "Retokenizing to sp8192 (parallel, 16 workers)..."
python3 retokenize.py \
    --input-dir ./data/datasets/fineweb10B_sp1024 \
    --output-dir ./data/datasets/fineweb10B_sp8192 \
    --old-tokenizer ./data/tokenizers/fineweb_1024_bpe.model \
    --new-tokenizer ./data/tokenizers/fineweb_8192_bpe.model \
    --workers 16
log "Retokenization complete."

RETOK_TRAIN=$(ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)
RETOK_VAL=$(ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin 2>/dev/null | wc -l)
log "Retokenized: $RETOK_TRAIN train, $RETOK_VAL val shards"

if [ "$RETOK_TRAIN" -lt 80 ] || [ "$RETOK_VAL" -lt 1 ]; then
    log "ERROR: Retokenization failed!"
    exit 1
fi

# --- STEP 5: Training (80 min cap for 1-GPU, stride=128) ---
log "Starting training on 1xH100..."
RUN_ID=submission_v1 \
MAX_WALLCLOCK_SECONDS=4800 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
log "Training complete."

# --- STEP 6: Results ---
log "=== RESULTS ==="
tail -30 logs/submission_v1.txt
grep -E "val_bpb|final_int8|submission size" logs/submission_v1.txt

# --- STEP 7: Optional re-eval with stride=64 ---
log "Re-evaluating with stride=64..."
EVAL_ONLY_MODEL=final_model.int8.ptz \
EVAL_STRIDE=64 \
RUN_ID=submission_v1_s64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
log "Stride-64 eval complete."
tail -5 logs/submission_v1_s64.txt

log "=== DONE ==="
ELAPSED=$(( $(date +%s) - START_TIME ))
log "Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
