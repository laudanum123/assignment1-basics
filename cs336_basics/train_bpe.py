
from collections import Counter
import regex

def create_vocabulary(vocab_size: int, special_tokens: list[bytes]):
    number_of_bytes = vocab_size if vocab_size <= 256 else 256
    base_vocabulary = [bytes([b]) for b in range(number_of_bytes)]
    vocabulary = base_vocabulary + special_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocabulary)} 
    return vocabulary, token_to_id

def pretokenize(input_path: str) -> Counter:
    text = read_txt_file(input_path)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Split corpus on special token <|endoftext|>
    parts = text.split("<|endoftext|>")

    pretoken_count = Counter()
    for part in parts:
        pretoken_count += Counter(match.group(0) for match in regex.finditer(PAT, part))

    return pretoken_count
    


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
        most_common_pair, _ = adjacent_pairs.most_common(1)[0]
        merges.append((most_common_pair[0], most_common_pair[1]))
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab.append(new_token)
        new_word_seqs = apply_merge(most_common_pair, word_seqs)


    
    return new_word_seqs, vocab, merges

def prettyfy_seqs_print(word_seqs):
    pretty_word_seqs = {w: [b.decode('latin1') for b in seq] for w, seq in word_seqs.items()}
    print(pretty_word_seqs)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    vocab, token_to_id = create_vocabulary(vocab_size, special_tokens)
    word_freq = pretokenize(input_path)
    word_seqs = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_freq}

    learned_word_seqs, vocab, merges = learn_bpe_merges(
        word_seqs=word_seqs, word_freq=word_freq, vocab=vocab, token_to_id=token_to_id, vocab_size=vocab_size
    )
    
    return vocab, merges
    prettyfy_seqs_print(learned_word_seqs)
    print(vocab)






def main():
    train_bpe(input_path='tests/fixtures/tinystories_sample.txt', vocab_size=500, special_tokens=["<|endoftext|>"])


if __name__ == "__main__":
    main()
