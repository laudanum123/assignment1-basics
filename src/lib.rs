use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{self, Write};

type ByteToken = Vec<u8>;
type Pair = (ByteToken, ByteToken);

/// A pair entry in the heap with its count
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    count: i64,
    pair: Pair,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by count (higher is better)
        match self.count.cmp(&other.count) {
            Ordering::Equal => {
                // Break ties lexicographically (larger pair is better)
                // Compare first element first, then second element (matches Python's max(tuple))
                match self.pair.0.cmp(&other.pair.0) {
                    Ordering::Equal => self.pair.1.cmp(&other.pair.1),
                    ord => ord,
                }
            }
            ord => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Extract adjacent pairs from a sequence
fn extract_pairs(seq: &[ByteToken], freq: i64) -> HashMap<Pair, i64> {
    let mut pairs = HashMap::default();
    for i in 0..seq.len().saturating_sub(1) {
        let pair = (seq[i].clone(), seq[i + 1].clone());
        *pairs.entry(pair).or_insert(0) += freq;
    }
    pairs
}

/// Merge a pair in a sequence
fn merge_pair_in_seq(seq: &[ByteToken], a: &[u8], b: &[u8]) -> Vec<ByteToken> {
    if seq.len() < 2 {
        return seq.to_vec();
    }

    let mut new_seq = Vec::with_capacity(seq.len());
    let mut i = 0;

    while i < seq.len() {
        if i < seq.len() - 1 && seq[i].as_slice() == a && seq[i + 1].as_slice() == b {
            let mut merged = a.to_vec();
            merged.extend_from_slice(b);
            new_seq.push(merged);
            i += 2;
        } else {
            new_seq.push(seq[i].clone());
            i += 1;
        }
    }

    new_seq
}

/// BPE trainer with heap-based optimization
struct HeapIncrementalBPE {
    word_seqs: HashMap<String, Vec<ByteToken>>,
    word_freq: HashMap<String, i64>,
    pair_counts: HashMap<Pair, i64>,
    heap: BinaryHeap<HeapEntry>,
    token_to_words: HashMap<ByteToken, HashSet<String>>,
}

impl HeapIncrementalBPE {
    fn new(word_seqs: HashMap<String, Vec<ByteToken>>, word_freq: HashMap<String, i64>) -> Self {
        let mut pair_counts = HashMap::default();
        let mut token_to_words: HashMap<ByteToken, HashSet<String>> = HashMap::default();

        // Build token-to-words index and initial pair counts
        for (word, seq) in &word_seqs {
            for token in seq {
                token_to_words
                    .entry(token.clone())
                    .or_default()
                    .insert(word.clone());
            }

            if seq.len() > 1 {
                let freq = *word_freq.get(word).unwrap_or(&0);
                for i in 0..seq.len() - 1 {
                    let pair = (seq[i].clone(), seq[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }
        }

        // Build initial heap
        let mut heap = BinaryHeap::with_capacity(pair_counts.len());
        for (pair, count) in &pair_counts {
            heap.push(HeapEntry {
                count: *count,
                pair: pair.clone(),
            });
        }

        Self {
            word_seqs,
            word_freq,
            pair_counts,
            heap,
            token_to_words,
        }
    }

    fn get_most_common_pair(&mut self) -> Option<Pair> {
        // Get all pairs and their counts, find the maximum
        let mut max_count = 0;
        for count in self.pair_counts.values() {
            if *count > max_count {
                max_count = *count;
            }
        }

        if max_count == 0 {
            return None;
        }

        // Collect all pairs with maximum count
        let mut candidates: Vec<Pair> = self.pair_counts
            .iter()
            .filter(|(_, &count)| count == max_count)
            .map(|(pair, _)| pair.clone())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort candidates and return lexicographically largest
        // This matches Python's max(tuple) behavior
        candidates.sort_by(|a, b| match b.0.cmp(&a.0) {
            Ordering::Equal => b.1.cmp(&a.1),
            ord => ord,
        });

        Some(candidates[0].clone())
    }

    fn apply_merge(&mut self, pair: &Pair) -> ByteToken {
        let (a, b) = pair;
        let mut new_token = a.clone();
        new_token.extend_from_slice(b);

        // Find affected words
        let candidate_words: Vec<String> = self
            .token_to_words
            .get(a)
            .map(|words| words.iter().cloned().collect())
            .unwrap_or_default();

        let mut affected_words = Vec::new();
        for word in candidate_words {
            let seq = &self.word_seqs[&word];
            if seq.len() < 2 {
                continue;
            }

            // Check if pair exists
            let has_pair = (0..seq.len() - 1)
                .any(|i| seq[i].as_slice() == a && seq[i + 1].as_slice() == b);

            if has_pair {
                affected_words.push(word);
            }
        }

        // Update affected words
        let mut changed_pairs = HashSet::default();
        for word in affected_words {
            let old_seq = self.word_seqs[&word].clone();
            let freq = *self.word_freq.get(&word).unwrap_or(&0);

            // Remove old pairs
            let old_pairs = extract_pairs(&old_seq, freq);
            for (pair, count) in old_pairs {
                *self.pair_counts.entry(pair.clone()).or_insert(0) -= count;
                changed_pairs.insert(pair);
            }

            // Apply merge
            let new_seq = merge_pair_in_seq(&old_seq, a, b);
            self.word_seqs.insert(word.clone(), new_seq.clone());

            // Add new pairs
            let new_pairs = extract_pairs(&new_seq, freq);
            for (pair, count) in new_pairs {
                *self.pair_counts.entry(pair.clone()).or_insert(0) += count;
                changed_pairs.insert(pair);
            }

            // Update token index
            for token in &old_seq {
                if !new_seq.contains(token) {
                    if let Some(words) = self.token_to_words.get_mut(token) {
                        words.remove(&word);
                    }
                }
            }
            for token in &new_seq {
                self.token_to_words
                    .entry(token.clone())
                    .or_default()
                    .insert(word.clone());
            }
        }

        // Add updated pairs to heap
        for pair in changed_pairs {
            let count = *self.pair_counts.get(&pair).unwrap_or(&0);
            if count > 0 {
                self.heap.push(HeapEntry { count, pair });
            }
        }

        new_token
    }
}

/// Train BPE from Rust (accepts pretokenized word frequencies from Python)
#[pyfunction]
#[pyo3(signature = (word_freq_dict, vocab_size, special_tokens))]
fn train_bpe_rust<'py>(
    py: Python<'py>,
    word_freq_dict: &Bound<'py, PyDict>,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<Bound<'py, PyTuple>> {
    // Convert Python dict to Rust HashMap
    println!("Converting word frequencies from Python...");
    io::stdout().flush().ok();
    
    let mut word_freq = HashMap::default();
    for (key, value) in word_freq_dict.iter() {
        let word: String = key.extract()?;
        let freq: i64 = value.extract()?;
        word_freq.insert(word, freq);
    }
    
    println!("Received {} unique tokens from pretokenization", word_freq.len());
    io::stdout().flush().ok();

    // Create initial vocabulary
    let mut vocab: Vec<ByteToken> = (0..256).map(|b| vec![b as u8]).collect();
    for token in &special_tokens {
        vocab.push(token.as_bytes().to_vec());
    }

    // Initialize word sequences
    let mut word_seqs = HashMap::default();
    for (word, _) in &word_freq {
        let seq: Vec<ByteToken> = word.as_bytes().iter().map(|&b| vec![b]).collect();
        word_seqs.insert(word.clone(), seq);
    }

    // Convert word_freq to i64
    let word_freq_i64: HashMap<String, i64> = word_freq
        .into_iter()
        .map(|(k, v)| (k, v as i64))
        .collect();

    // Learn merges
    let mut bpe = HeapIncrementalBPE::new(word_seqs, word_freq_i64);
    let mut merges = Vec::new();

    println!("Starting BPE training: target vocab size = {}", vocab_size);
    io::stdout().flush().ok();
    
    while vocab.len() < vocab_size {
        if let Some(pair) = bpe.get_most_common_pair() {
            merges.push((pair.0.clone(), pair.1.clone()));
            let new_token = bpe.apply_merge(&pair);
            vocab.push(new_token);
            
            // Print progress every 50 tokens
            if vocab.len() % 50 == 0 {
                println!("Progress: {}/{} tokens ({:.1}%)", 
                    vocab.len(), vocab_size, 
                    (vocab.len() as f64 / vocab_size as f64) * 100.0);
                io::stdout().flush().ok();
            }
        } else {
            break;
        }
    }
    
    println!("BPE training complete: final vocab size = {}", vocab.len());
    io::stdout().flush().ok();

    // Build id_to_token map as Python dict
    let id_to_token_py = PyDict::new_bound(py);
    for (i, token) in vocab.into_iter().enumerate() {
        let token_list = PyList::new_bound(py, token.iter());
        id_to_token_py.set_item(i, token_list)?;
    }

    // Build merges as Python list
    let merges_py = PyList::empty_bound(py);
    for (a, b) in merges {
        let a_list = PyList::new_bound(py, a.iter());
        let b_list = PyList::new_bound(py, b.iter());
        let pair = PyTuple::new_bound(py, &[a_list.as_any(), b_list.as_any()]);
        merges_py.append(pair)?;
    }

    Ok(PyTuple::new_bound(py, &[id_to_token_py.as_any(), merges_py.as_any()]))
}

/// Python module
#[pymodule]
fn cs336_bpe_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_rust, m)?)?;
    Ok(())
}
