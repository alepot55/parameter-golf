"""Retokenize binary shards between SentencePiece vocabularies — parallel version."""

import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece as spm

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
CHUNK_TOKENS = 500_000


def retokenize_shard(args_tuple):
    """Process a single shard. Each worker loads its own tokenizer instances."""
    src_path, dst_path, old_tok_path, new_tok_path = args_tuple
    sp_old = spm.SentencePieceProcessor(model_file=old_tok_path)
    sp_new = spm.SentencePieceProcessor(model_file=new_tok_path)

    name = os.path.basename(src_path)
    header = np.fromfile(src_path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset_bytes = 256 * 4

    new_tokens_list = []
    processed = 0
    while processed < num_tokens:
        n = min(CHUNK_TOKENS, num_tokens - processed)
        chunk = np.fromfile(src_path, dtype="<u2", count=n,
                            offset=offset_bytes + processed * 2)
        text = sp_old.decode(chunk.tolist())
        new_toks = sp_new.encode(text)
        new_tokens_list.append(np.array(new_toks, dtype=np.uint16))
        processed += n

    all_new = np.concatenate(new_tokens_list)

    out_header = np.zeros(256, dtype="<i4")
    out_header[0] = SHARD_MAGIC
    out_header[1] = SHARD_VERSION
    out_header[2] = len(all_new)
    with open(dst_path, "wb") as f:
        f.write(out_header.tobytes())
        f.write(all_new.tobytes())

    ratio = num_tokens / len(all_new) if all_new.size else 0
    print(f"  {name}: {num_tokens} -> {len(all_new)} tokens ({ratio:.2f}x)", flush=True)
    return num_tokens, len(all_new)


def main():
    parser = argparse.ArgumentParser(description="Retokenize binary shards between vocabularies")
    parser.add_argument("--input-dir", default=os.environ.get("SRC_DIR", "./data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--output-dir", default=os.environ.get("DST_DIR", "./data/datasets/fineweb10B_sp8192"))
    parser.add_argument("--old-tokenizer", default=os.environ.get("SP_OLD", "./data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--new-tokenizer", default=os.environ.get("SP_NEW", "./data/tokenizers/fineweb_8192_bpe.model"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = parser.parse_args()

    if not os.path.exists(args.new_tokenizer):
        print(f"ERROR: {args.new_tokenizer} not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    val_shards = sorted(glob.glob(os.path.join(args.input_dir, "fineweb_val_*.bin")))
    train_shards = sorted(glob.glob(os.path.join(args.input_dir, "fineweb_train_*.bin")))
    all_shards = val_shards + train_shards
    print(f"Found {len(val_shards)} val, {len(train_shards)} train shards")

    if not all_shards:
        print(f"ERROR: No shards found in {args.input_dir}")
        sys.exit(1)

    shard_args = [
        (src, os.path.join(args.output_dir, os.path.basename(src)),
         args.old_tokenizer, args.new_tokenizer)
        for src in all_shards
    ]

    workers = min(args.workers, len(shard_args))
    print(f"Retokenizing {len(shard_args)} shards with {workers} workers...")

    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(retokenize_shard, shard_args)
    else:
        results = [retokenize_shard(a) for a in shard_args]

    total_old = sum(r[0] for r in results)
    total_new = sum(r[1] for r in results)
    print(f"\nDone! {total_old} -> {total_new} tokens ({total_old/max(total_new,1):.2f}x)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
