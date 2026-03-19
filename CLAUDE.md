# Parameter Golf — Competition Codebase

## Project Overview
OpenAI Parameter Golf Challenge: train the best LM fitting in a 16MB artifact (code + int8 zlib-compressed weights) in ≤10 min on 8×H100s. Metric: val_bpb (bits per byte) on FineWeb validation (first 50K docs). Baseline: 1.2244 bpb. Target: beat SOTA by ≥0.005 nats with p<0.01.

## Repository Structure
```
parameter-golf/
├── train_gpt.py          # Main training script (counts toward 16MB artifact)
├── train_gpt_mlx.py      # Mac local dev script (not for submission)
├── data/
│   ├── cached_challenge_fineweb.py   # Dataset download script
│   ├── tokenizers/                    # SentencePiece .model files
│   └── datasets/fineweb10B_sp1024/   # Tokenized shards
├── records/
│   ├── track_10min_16mb/             # SOTA record submissions
│   └── track_non_record_16mb/        # Non-record / unlimited submissions
├── scripts/                           # Utility scripts
└── requirements.txt
```

## Key Constraints
- **Artifact size**: code bytes + int8 zlib-compressed weights ≤ 16,000,000 bytes (decimal)
- **Training time**: ≤ 10 min wallclock on 8×H100 SXM via torchrun
- **Evaluation time**: ≤ 10 min on 8×H100 (any strategy, any seq_len)
- **No network access during eval**: artifact must be self-contained
- **Metric**: val_bpb = loss_nats / (bytes_per_token × ln(2)), tokenizer-agnostic
- **SOTA threshold**: ≥0.005 nats improvement, p<0.01 across multiple runs
- **Tokenizer changes**: require proof of correct bpb calculation, heavily scrutinized

## Development Workflow
- **Local dev**: RTX 4070 (single GPU, `nproc_per_node=1`) — iterate cheap, validate ideas
- **Final submission**: 8×H100 on RunPod — 10-min constrained runs
- Local runs use `MAX_WALLCLOCK_SECONDS=0` for unlimited time
- Use `VAL_LOSS_EVERY=200` for validation during development
- Use `--train-shards 1` for small data subset during iteration

## Running Commands
```bash
# Download data (small subset for dev)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Local 1-GPU training (dev)
RUN_ID=experiment_name \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=200 \
ITERATIONS=2000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8×H100 final submission run
RUN_ID=submission_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture (Current Baseline)
- 9 transformer blocks, d_model=512, 8 heads / 4 KV heads (GQA)
- relu² MLP with 2× expansion (hidden=1024)
- RoPE (base=10000), QK-norm (RMS on Q,K), tanh logit softcap (30)
- U-Net skip connections (encoder/decoder halves with learned skip_weights)
- x0 residual mixing (resid_mix per block)
- Tied input/output embeddings, vocab=1024 SentencePiece BPE
- ~3.2M parameters → compressed ~3MB

## Optimizer Setup (Current Baseline)
- **Muon** (Newton-Schulz 5-step): 2D matrix params in transformer blocks
  - lr=0.04, momentum=0.95, nesterov=True
- **Adam**: embeddings (lr=0.05 tied), scalars/vectors (lr=0.04)
  - betas=(0.9, 0.95), eps=1e-8
- WSD schedule: warmdown over last 1200 iters
- Warmup: 20 steps (compile warmup, state reset after)

## Implementation Priority (Competition Strategy)

### Phase 1 — Vocab + Architecture [HIGH IMPACT]
1. **Vocab 1024→4096** with BBPE or SentencePiece trained on FineWeb
   - ~29% bpb improvement from compression alone
   - Cost: +1.5M embedding params (tied), worth it
   - RISK: must prove correct bpb calculation
2. **Depth recurrence**: 3 unique blocks × 3 loops = 9 effective layers
   - Saves ~2M params → reinvest in d_model or more unique blocks
   - Add per-loop LoRA adapters (rank 32) to differentiate iterations
   - Sandwich pattern: unique first/last, shared middle

### Phase 2 — Training Efficiency [MEDIUM IMPACT]
3. **Sequence length curriculum**: 256→512→1024→2048 across training
4. **SwiGLU** (d_ff = 8d/3 ≈ 1365) instead of relu² (better loss/param)
5. **L1 regularization** (λ≈1e-4) for weight compressibility
6. **OT regularization** (λ≈0.01) for smooth hidden trajectories
7. **FP8 matmuls** via torchao on H100 (~47% throughput boost)

### Phase 3 — Compression [MEDIUM IMPACT]
8. **K-means weight clustering** (16-32 centroids) + bitpacked indices
9. **Weight serialization order** optimization for zlib
10. Custom encoding in code artifact (arithmetic/ANS) if budget allows

### Phase 4 — Evaluation [HIGH RISK/REWARD]
11. **Eval at longer seq_len** (2048-4096) with YaRN RoPE extension
12. **Test-time training**: 1-3 gradient steps on last 25% MLPs per doc
13. **Checkpoint ensemble** (average logits from multiple checkpoints)

## Parameter Budget Model
```
Target: ~15-20M stored params → 12-15MB compressed → room for code
With depth recurrence (3 unique × 3 loops):
  - 3 unique blocks × ~3.15M/block = 9.45M
  - Tied embeddings V=4096: 4096 × 512 = 2.1M
  - Per-loop LoRA (rank 32): ~0.9M
  - Skip weights, norms, gains: ~0.1M
  Total stored: ~12.5M params → ~5-8MB int8+zlib compressed
```

## Code Style & Constraints
- train_gpt.py must be <1500 lines (hard repo limit)
- All code must be in train_gpt.py (counts toward artifact)
- MIT license required
- Must run with `torchrun` on 8×H100
- Python 3.10+, PyTorch 2.x, CUDA
- Dependencies: torch, numpy, sentencepiece, huggingface-hub, datasets, tqdm

## Artifact Size Calculation
```python
# Size = len(code_bytes) + len(zlib.compress(int8_weights_bytes))
code_bytes = open("train_gpt.py", "rb").read()
# After training: model is int8 quantized + zlib compressed
# Total must be ≤ 16,000,000 bytes
```

## Key Technical Decisions Log
<!-- Track important decisions and their rationale here -->
- [BASELINE] relu² MLP, GQA 8/4, RoPE, tied embeddings V=1024
- [TODO] Evaluate vocab increase ROI
- [TODO] Test depth recurrence penalty at this scale

## References
- Baseline repo: github.com/openai/parameter-golf
- NanoGPT speedrun: github.com/KellerJordan/modded-nanogpt
- Muon: github.com/KellerJordan/Muon
- modded-nanogpt speedrun analysis: lesswrong.com/posts/j3gp8tebQiFJqzBgg
