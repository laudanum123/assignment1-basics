#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all BPE implementations including Rust.
"""
import time
import sys
from cs336_basics.train_bpe import train_bpe as train_bpe_original
from cs336_basics.train_bpe_heap import train_bpe as train_bpe_heap
from cs336_basics.train_bpe_truly_incremental import train_bpe as train_bpe_incremental

try:
    from cs336_bpe_rust import train_bpe_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠ Warning: Rust implementation not available")


def benchmark_implementation(name, train_func, input_path, vocab_size, special_tokens, runs=3):
    """Run the BPE training multiple times and report statistics."""
    times = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.perf_counter()
        
        if name == "Rust Implementation":
            id_to_token_raw, merges_raw = train_func(input_path, vocab_size, special_tokens)
            # Convert to match Python format
            id_to_token = {k: bytes(v) for k, v in id_to_token_raw.items()}
            merges = [(bytes(a), bytes(b)) for a, b in merges_raw]
        else:
            id_to_token, merges = train_func(input_path, vocab_size, special_tokens)
        
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"{elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Min:     {min_time:.4f}s")
    print(f"  Max:     {max_time:.4f}s")
    print(f"  Vocab:   {len(id_to_token)} tokens")
    print(f"  Merges:  {len(merges)} merges")
    
    return {
        'name': name,
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'vocab_size': len(id_to_token),
        'merges': len(merges)
    }


def main():
    # Configuration
    input_path = 'tests/fixtures/corpus.en'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    runs = 3
    
    print(f"{'='*60}")
    print(f"BPE Implementation Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Input: {input_path}")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Runs per implementation: {runs}")
    
    results = []
    
    # Benchmark Python implementations
    results.append(benchmark_implementation(
        "Python Original",
        train_bpe_original,
        input_path,
        vocab_size,
        special_tokens,
        runs
    ))
    
    results.append(benchmark_implementation(
        "Python Heap",
        train_bpe_heap,
        input_path,
        vocab_size,
        special_tokens,
        runs
    ))
    
    results.append(benchmark_implementation(
        "Python Truly Incremental",
        train_bpe_incremental,
        input_path,
        vocab_size,
        special_tokens,
        runs
    ))
    
    # Benchmark Rust if available
    if RUST_AVAILABLE:
        results.append(benchmark_implementation(
            "Rust Implementation",
            train_bpe_rust,
            input_path,
            vocab_size,
            special_tokens,
            runs
        ))
    
    # Compare results
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    print("\nAverage times:")
    for res in results:
        print(f"  {res['name']:30} {res['avg']:.4f}s")
    
    print("\nSpeedup vs Original:")
    baseline = results[0]['avg']
    for res in results[1:]:
        speedup = baseline / res['avg']
        percent_faster = ((baseline - res['avg']) / baseline) * 100
        if res['avg'] < baseline:
            print(f"  {res['name']:30} {speedup:.2f}x faster ({percent_faster:.1f}% improvement)")
        else:
            slowdown = res['avg'] / baseline
            percent_slower = ((res['avg'] - baseline) / baseline) * 100
            print(f"  {res['name']:30} {slowdown:.2f}x SLOWER ({percent_slower:.1f}% worse)")
    
    if RUST_AVAILABLE:
        print("\nRust vs Best Python:")
        rust_time = results[-1]['avg']
        best_python_time = min(r['avg'] for r in results[:-1])
        speedup = best_python_time / rust_time
        percent_faster = ((best_python_time - rust_time) / best_python_time) * 100
        print(f"  Rust is {speedup:.2f}x faster than best Python ({percent_faster:.1f}% improvement)")
    
    # Visual performance comparison
    print("\n" + "="*60)
    print("VISUAL COMPARISON")
    print("="*60)
    max_time = max(r['avg'] for r in results)
    for res in results:
        bar_length = int((res['avg'] / max_time) * 40)
        bar = "█" * bar_length
        print(f"{res['name']:30} {bar} {res['avg']:.4f}s")
    
    # Verify correctness
    print("\n" + "="*60)
    print("CORRECTNESS CHECK")
    print("="*60)
    if all(r['vocab_size'] == results[0]['vocab_size'] and 
           r['merges'] == results[0]['merges'] for r in results):
        print(f"✓ All implementations produce the same vocab size and number of merges")
    else:
        print(f"⚠ WARNING: Implementations produce different results!")
        for res in results:
            print(f"  {res['name']:30} vocab={res['vocab_size']}, merges={res['merges']}")


if __name__ == "__main__":
    main()
