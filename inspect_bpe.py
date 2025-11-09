#!/usr/bin/env python3
"""
Utility to inspect and understand BPE vocabulary and merges.
"""
import json
import sys


def load_vocab(vocab_path: str) -> dict[int, bytes]:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        vocab_raw = json.load(f)
    return {int(k): bytes(v) for k, v in vocab_raw.items()}


def load_merges(merges_path: str) -> list[tuple[bytes, bytes]]:
    """Load merges from text file."""
    merges = []
    with open(merges_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                a_hex, b_hex = line.split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
    return merges


def print_token(token_id: int, token_bytes: bytes):
    """Pretty print a token."""
    try:
        token_str = token_bytes.decode('utf-8')
        print(f"  ID {token_id:5d}: {repr(token_str):30s} (bytes: {token_bytes.hex()})")
    except UnicodeDecodeError:
        print(f"  ID {token_id:5d}: <non-UTF8: {token_bytes.hex()}>")


def print_merge(idx: int, a: bytes, b: bytes):
    """Pretty print a merge rule."""
    try:
        a_str = a.decode('utf-8')
        b_str = b.decode('utf-8')
        print(f"  {idx:5d}: {repr(a_str):20s} + {repr(b_str):20s}")
    except UnicodeDecodeError:
        print(f"  {idx:5d}: <{a.hex()}> + <{b.hex()}>")


def main():
    vocab_path = sys.argv[1] if len(sys.argv) > 1 else "rust_bpe_vocab.json"
    merges_path = sys.argv[2] if len(sys.argv) > 2 else "rust_bpe_merges.txt"
    
    print("Loading BPE files...")
    vocab = load_vocab(vocab_path)
    merges = load_merges(merges_path)
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY: {len(vocab)} tokens")
    print(f"{'='*80}")
    
    # Show first 20 tokens
    print("\nFirst 20 tokens:")
    for token_id in sorted(vocab.keys())[:20]:
        print_token(token_id, vocab[token_id])
    
    # Show last 20 tokens
    print("\nLast 20 tokens:")
    for token_id in sorted(vocab.keys())[-20:]:
        print_token(token_id, vocab[token_id])
    
    # Show some interesting tokens (if they exist)
    print("\nSpecial/Common tokens:")
    for token_id in sorted(vocab.keys()):
        token_bytes = vocab[token_id]
        try:
            token_str = token_bytes.decode('utf-8')
            if token_str in ['<|endoftext|>', ' ', '\n', 'the', 'The', 'and', 'a']:
                print_token(token_id, token_bytes)
        except UnicodeDecodeError:
            pass
    
    print(f"\n{'='*80}")
    print(f"MERGES: {len(merges)} rules")
    print(f"{'='*80}")
    
    # Show first 20 merges
    print("\nFirst 20 merge rules:")
    for i, (a, b) in enumerate(merges[:20], 1):
        print_merge(i, a, b)
    
    # Show last 20 merges
    print("\nLast 20 merge rules:")
    for i, (a, b) in enumerate(merges[-20:], len(merges) - 19):
        print_merge(i, a, b)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Number of merges: {len(merges)}")
    print(f"  - Base vocabulary size: {len(vocab) - len(merges)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
