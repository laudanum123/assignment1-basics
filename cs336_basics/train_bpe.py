
from collections import Counter
import regex
import os
from multiprocessing import Pool, cpu_count

TOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _count_tokens_in_text(text: str) -> Counter:
    """Run the regex tokenizer over the text and return token counts."""
    parts = text.split("<|endoftext|>")
    pretoken_count = Counter()
    for part in parts:
        pretoken_count += Counter(match.group(0) for match in regex.finditer(TOKEN_PATTERN, part))
    return pretoken_count

def create_vocabulary(vocab_size: int, special_tokens: list[bytes]):
    number_of_bytes = vocab_size if vocab_size <= 256 else 256
    base_vocabulary = [bytes([b]) for b in range(number_of_bytes)]
    vocabulary = base_vocabulary + special_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocabulary)} 
    return vocabulary, token_to_id

def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    with open(file_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


def pretokenize_chunk(args: tuple[str, int, int, list[bytes]]) -> Counter:
    """
    Pretokenize a chunk of the file from start to end position.
    
    Args:
        args: Tuple of (file_path, start, end, special_tokens)
    
    Returns:
        Counter with pretokenized words and their counts
    """
    file_path, start, end, special_tokens = args
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    return _count_tokens_in_text(chunk)


def pretokenize(
    input_path: str,
    special_tokens: list[bytes],
    num_processes: int = None,
    chunks_per_process: int | None = None,
) -> Counter:
    """
    Pretokenize the input file using multiprocessing.
    
    Args:
    input_path: Path to the input text file
    special_tokens: List of special token bytes
    num_processes: Number of processes to use (default: cpu_count()-2, min 1, max 8)
    chunks_per_process: Multiplier controlling how many chunks to create per process (default: 2 when multiprocessing)
    
    Returns:
        Counter with pretokenized words and their counts
    """
    if num_processes is None:
        # Use cpu_count()-2 to leave some resources for the system
        # but ensure we have at least 1 process and cap at 8 to avoid overload
        num_processes = max(1, min(cpu_count() - 2, 8))
    else:
        # Ensure user-provided value is reasonable
        num_processes = max(1, min(num_processes, 8))
    
    if chunks_per_process is None:
        chunks_per_process = 2 if num_processes > 1 else 1
    chunks_per_process = max(1, chunks_per_process)

    # For small files or single process, use the simple approach
    if num_processes <= 1:
        text = read_txt_file(input_path)
        return _count_tokens_in_text(text)
    
    # Determine how many chunks to request and find boundaries at special token positions
    requested_chunks = max(num_processes, num_processes * chunks_per_process)
    split_token = b"<|endoftext|>"
    boundaries = find_chunk_boundaries(input_path, requested_chunks, split_token)
    
    # Create arguments for each chunk
    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    total_chunks = len(chunk_args)
    if total_chunks == 0:
        text = read_txt_file(input_path)
        return _count_tokens_in_text(text)

    chunk_msg = (
        f"[Pretokenize] Requested {requested_chunks} chunk(s); "
        f"created {total_chunks} chunk(s) to run across {num_processes} process(es)."
    )
    if total_chunks < requested_chunks:
        chunk_msg += " (Some boundaries merged because the split token was not found nearby.)"
    print(chunk_msg, flush=True)
    print(
        f"[Pretokenize] Processing {total_chunks} chunk(s) across {num_processes} process(es)...",
        flush=True,
    )

    final_count = Counter()
    with Pool(processes=num_processes) as pool:
        for completed, count in enumerate(
            pool.imap_unordered(pretokenize_chunk, chunk_args), start=1
        ):
            final_count += count
            percent = (completed / total_chunks) * 100
            print(
                f"[Pretokenize] Finished chunk {completed}/{total_chunks} ({percent:.1f}%)",
                flush=True,
            )
    
    return final_count
    
def read_txt_file(input_path: str) -> str:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def count_adjacent_pairs(word_seqs, word_freq):
    counts = Counter()
    for w, seq in word_seqs.items():
        for i in range(len(seq)-1):
            counts[(seq[i], seq[i+1])] += word_freq[w]
    return counts

def merge_pair_in_seq(seq: list[bytes], a: bytes, b: bytes) -> list[bytes]:
    """Return a new sequence where each non-overlapping (a,b) is replaced by a + b"""
    new_seq = []
    i = 0
    while  i < len(seq):
        if i < len(seq) -1 and (seq[i], seq[i+1]) == (a, b):
            new_seq.append(a + b)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


def apply_merge(best_pair: tuple[bytes, bytes], word_seqs: dict[str, list[bytes]]) -> dict[str, list[bytes]]:
    """Apply a single merge operation to all word sequences."""
    for word, seq in word_seqs.items():
        new_seq = merge_pair_in_seq(seq, a=best_pair[0], b=best_pair[1])
        word_seqs[word] = new_seq
    return word_seqs


def learn_bpe_merges(
        word_seqs: dict[str, list[bytes]],
        word_freq: Counter,
        vocab: list[bytes],
        token_to_id: dict[bytes, int],
        vocab_size: int
) -> list[tuple[bytes, bytes]]:
    
    merges = []
    while len(vocab) < vocab_size:
        adjacent_pairs = count_adjacent_pairs(word_seqs=word_seqs, word_freq=word_freq)
        
        # Get the maximum frequency
        max_freq = max(adjacent_pairs.values())

        # Get all pairs with maximum frequency
        top_pairs = [pair for pair, freq in adjacent_pairs.items() if freq == max_freq]

        most_common_pair = max(top_pairs)


        merges.append((most_common_pair[0], most_common_pair[1]))
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab.append(new_token)
        word_seqs = apply_merge(most_common_pair, word_seqs)

    return word_seqs, vocab, merges

def prettyfy_seqs_print(word_seqs):
    """Helper for pretty printing word sequences"""
    pretty_word_seqs = {w: [b.decode('latin1') for b in seq] for w, seq in word_seqs.items()}
    print(pretty_word_seqs)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int | None = None,
    chunks_per_process: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    vocab, token_to_id = create_vocabulary(vocab_size, special_tokens_bytes)
    word_freq = pretokenize(
        input_path,
        special_tokens=special_tokens_bytes,
        num_processes=num_processes,
        chunks_per_process=chunks_per_process,
    )
    word_seqs = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_freq}

    learned_word_seqs, vocab, merges = learn_bpe_merges(
        word_seqs=word_seqs, word_freq=word_freq, vocab=vocab, token_to_id=token_to_id, vocab_size=vocab_size
    )

    #prettyfy_seqs_print(learned_word_seqs)
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    #print(vocab)
    return id_to_token, merges


def main():
    train_bpe(input_path='tests/fixtures/corpus.en', vocab_size=500, special_tokens=["<|endoftext|>"])


if __name__ == "__main__":
    main()
