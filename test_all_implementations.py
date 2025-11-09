#!/usr/bin/env python3
"""
Test truly incremental implementation
"""
import time
from cs336_basics.train_bpe_optimized import train_bpe as train_bpe_optimized
from cs336_basics.train_bpe_incremental import train_bpe as train_bpe_incremental
from cs336_basics.train_bpe_truly_incremental import train_bpe as train_bpe_truly_incremental
from cs336_basics.train_bpe_heap import train_bpe as train_bpe_heap

def benchmark(name, func, input_path, vocab_size, special_tokens, runs=3):
    print(f"\n{name}")
    print("-" * 60)
    times = []
    for i in range(runs):
        print(f"Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.perf_counter()
        id_to_token, merges = func(input_path, vocab_size, special_tokens)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{elapsed:.3f}s")
    
    avg = sum(times) / len(times)
    print(f"Average: {avg:.3f}s")
    return avg, len(id_to_token), len(merges)

# Test on small corpus
print("="*60)
print("BENCHMARK: All Implementations")
print("="*60)

input_path = 'tests/fixtures/corpus.en'
vocab_size = 500
special_tokens = ["<|endoftext|>"]

opt_time, opt_vocab, opt_merges = benchmark(
    "Optimized",
    train_bpe_optimized,
    input_path, vocab_size, special_tokens
)

inc_time, inc_vocab, inc_merges = benchmark(
    "Incremental (with full recount)",
    train_bpe_incremental,
    input_path, vocab_size, special_tokens
)

true_inc_time, true_inc_vocab, true_inc_merges = benchmark(
    "Truly Incremental (delta updates only)",
    train_bpe_truly_incremental,
    input_path, vocab_size, special_tokens
)

heap_time, heap_vocab, heap_merges = benchmark(
    "Heap-based (incremental + priority queue)",
    train_bpe_heap,
    input_path, vocab_size, special_tokens
)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Optimized:          {opt_time:.3f}s (1.00x baseline)")
print(f"Incremental:        {inc_time:.3f}s ({opt_time/inc_time:.2f}x speedup)")
print(f"Truly Incremental:  {true_inc_time:.3f}s ({opt_time/true_inc_time:.2f}x speedup)")
print(f"Heap-based:         {heap_time:.3f}s ({opt_time/heap_time:.2f}x speedup)")

print(f"\n{'='*60}")
print("SPEEDUP COMPARISON")
print(f"{'='*60}")

times = [
    ("Optimized", opt_time),
    ("Incremental", inc_time),
    ("Truly Incremental", true_inc_time),
    ("Heap-based", heap_time)
]
times.sort(key=lambda x: x[1])

print(f"\nRanking (fastest to slowest):")
for i, (name, t) in enumerate(times, 1):
    print(f"  {i}. {name:20s} {t:.3f}s")

best_name, best_time = times[0]
print(f"\n⭐ Winner: {best_name}")

# Verify correctness
if (opt_vocab == inc_vocab == true_inc_vocab == heap_vocab and 
    opt_merges == inc_merges == true_inc_merges == heap_merges):
    print(f"\n✓ All implementations produce identical results")
else:
    print(f"\n⚠ WARNING: Different results!")
    print(f"  Optimized:         vocab={opt_vocab}, merges={opt_merges}")
    print(f"  Incremental:       vocab={inc_vocab}, merges={inc_merges}")
    print(f"  Truly Incremental: vocab={true_inc_vocab}, merges={true_inc_merges}")
    print(f"  Heap-based:        vocab={heap_vocab}, merges={heap_merges}")
