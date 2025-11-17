"""
Rust-accelerated BPE training with Python wrapper.

This provides a drop-in replacement for the Python BPE implementations
but with 2-3x performance improvement thanks to Rust.

Note: Pretokenization is done in Python using the original train_bpe.py code,
while the BPE learning algorithm runs in Rust for performance.
"""
import json
import time
from cs336_basics.train_bpe import pretokenize


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    chunks_per_process: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE using Rust implementation with Python pretokenization.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
    num_processes: Number of processes for pretokenization (default: auto-detect, max 8)
    chunks_per_process: How many chunks to request per process (default: auto, >=1)
    
    Returns:
        Tuple of (id_to_token dict, list of merges)
    """
    try:
        from cs336_basics.cs336_bpe_rust import train_bpe_rust
        
        # Step 1: Pretokenize using Python (from train_bpe.py)
        print(f"Pretokenizing {input_path} using Python (num_processes={num_processes or 'auto'})...")
        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        word_freq = pretokenize(
            input_path,
            special_tokens=special_tokens_bytes,
            num_processes=num_processes,
            chunks_per_process=chunks_per_process,
        )
        print(f"Pretokenization complete: {len(word_freq)} unique tokens")
        
        # Convert Counter to dict with string keys and int values
        word_freq_dict = {word: int(count) for word, count in word_freq.items()}
        
        # Step 2: Call Rust implementation for BPE learning
        print(f"Starting BPE learning in Rust (target vocab size: {vocab_size})...")
        id_to_token_raw, merges_raw = train_bpe_rust(word_freq_dict, vocab_size, special_tokens)
        
        # Convert to expected Python types
        id_to_token = {k: bytes(v) for k, v in id_to_token_raw.items()}
        merges = [(bytes(a), bytes(b)) for a, b in merges_raw]
        
        return id_to_token, merges
    
    except ImportError:
        # Fallback to Python implementation if Rust not available
        print("Warning: Rust implementation not available, falling back to Python")
        from cs336_basics.train_bpe import train_bpe as train_bpe_python
        return train_bpe_python(
            input_path,
            vocab_size,
            special_tokens,
            num_processes=num_processes,
            chunks_per_process=chunks_per_process,
        )


def main():
    start_time = time.time()
    
    input_path = 'tests/fixtures/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 4  # Limit to 4 processes to avoid system overload
    chunks_per_process = 100  # Request more, smaller chunks per worker for better progress granularity
    
    print(f"Training BPE with vocab_size={vocab_size} on {input_path}...")
    id_to_token, merges = train_bpe(
        input_path,
        vocab_size,
        special_tokens,
        num_processes=num_processes,
        chunks_per_process=chunks_per_process,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Serialize vocabulary (machine-readable)
    vocab_output = "outputs/rust_bpe_vocab.json"
    vocab_serializable = {k: list(v) for k, v in id_to_token.items()}
    with open(vocab_output, 'w') as f:
        json.dump(vocab_serializable, f, indent=2)
    print(f"Vocabulary saved to {vocab_output}")
    
    # Serialize vocabulary (human-readable)
    vocab_readable = "outputs/rust_bpe_vocab_readable.txt"
    with open(vocab_readable, 'w', encoding='utf-8') as f:
        f.write(f"BPE Vocabulary ({len(id_to_token)} tokens)\n")
        f.write("=" * 80 + "\n\n")
        for token_id in sorted(id_to_token.keys()):
            token_bytes = id_to_token[token_id]
            # Try to decode as UTF-8, otherwise show hex
            try:
                token_str = token_bytes.decode('utf-8')
                # Escape special characters for readability
                token_repr = repr(token_str)
            except UnicodeDecodeError:
                token_repr = f"<hex:{token_bytes.hex()}>"
            f.write(f"{token_id:5d}: {token_repr}\n")
    print(f"Human-readable vocabulary saved to {vocab_readable}")
    
    # Serialize merges (machine-readable)
    merges_output = "outputs/rust_bpe_merges.txt"
    with open(merges_output, 'w') as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    print(f"Merges saved to {merges_output}")
    
    # Serialize merges (human-readable)
    merges_readable = "outputs/rust_bpe_merges_readable.txt"
    with open(merges_readable, 'w', encoding='utf-8') as f:
        f.write(f"BPE Merge Rules ({len(merges)} merges)\n")
        f.write("=" * 80 + "\n\n")
        for i, (a, b) in enumerate(merges, 1):
            # Try to decode as UTF-8
            try:
                a_str = repr(a.decode('utf-8'))
                b_str = repr(b.decode('utf-8'))
                f.write(f"{i:5d}: {a_str} + {b_str}\n")
            except UnicodeDecodeError:
                f.write(f"{i:5d}: <{a.hex()}> + <{b.hex()}>\n")
    print(f"Human-readable merges saved to {merges_readable}")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print(f"Vocabulary size: {len(id_to_token)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    main()
