"""Implementation of BPE training example according to Sennrich et al., 2015."""

from collections import Counter

corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

def create_vocabulary(vocab_size: int, special_tokens: list[bytes]):
    base_vocabulary = [bytes([b]) for b in range(vocab_size)]
    vocabulary = base_vocabulary + special_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocabulary)} 

    return vocabulary, token_to_id

def pretokenize(text: str):
    return text.split()

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
        num_merges: int
) -> list[tuple[bytes, bytes]]:
    
    for i in range(num_merges):
        adjacent_pairs = count_adjacent_pairs(word_seqs=word_seqs, word_freq=word_freq)
        most_common_pair, _ = adjacent_pairs.most_common(1)[0]
        new_word_seqs = apply_merge(most_common_pair, word_seqs)
    
    return new_word_seqs


def main():
    vocab_size = 256
    special_tokens = [b"<|endoftext|>"]

        
    words = pretokenize(corpus)
    word_freq = Counter(words)
    word_seqs = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_freq}
    vocab, token_to_id = create_vocabulary(vocab_size=vocab_size, special_tokens=special_tokens)

    print(word_freq)
    pretty_word_seqs = {w: [b.decode('latin1') for b in seq] for w, seq in word_seqs.items()}
    print(pretty_word_seqs)
    counts = count_adjacent_pairs(word_seqs, word_freq)
    pretty_counts = Counter({(a.decode('latin1'), b.decode('latin1')): count for (a, b), count in counts.items()})
    print(pretty_counts)

    print(learn_bpe_merges(word_seqs=word_seqs, word_freq=word_freq, vocab=vocab, token_to_id=token_to_id, num_merges=6))



    


if __name__ == "__main__":
    main()