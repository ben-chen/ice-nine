use anyhow::{Error, Result};
use base64::{engine::general_purpose, Engine};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

type Token = Arc<[u8]>;

type TokenId = u32;

type Bigram = (TokenId, TokenId);

fn bigram_to_u64(bigram: Bigram) -> u64 {
    let (a, b) = bigram;
    ((a as u64) << 32) | (b as u64)
}

fn u64_to_bigram(bigram: u64) -> Bigram {
    let a = (bigram >> 32) as TokenId;
    let b = (bigram & 0xFFFFFFFF) as TokenId;
    (a, b)
}

pub struct Tokenizer {
    pub id_to_token: Arc<[Token]>,
    pub token_to_id: HashMap<Token, TokenId>,
    pub max_token_len: usize,
}

impl Tokenizer {
    pub fn new(tokens: Arc<[Token]>) -> Self {
        let mut max_token_len = 0;
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| {
                max_token_len = std::cmp::max(max_token_len, token.len());
                (token.clone(), id as TokenId)
            })
            .collect();

        Self {
            id_to_token: tokens,
            token_to_id,
            max_token_len,
        }
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let tokens = load_vocab_json(path)?;
        let id_to_token = Arc::from(tokens);
        Ok(Self::new(id_to_token))
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        save_vocab_json(&self.id_to_token, path)
    }

    pub fn split_into_tokens(&self, text: &str) -> Result<Vec<Token>> {
        let byte_text: Arc<_> = text.bytes().collect();

        let mut tokens = vec![];
        let mut start_idx = 0;
        'first_idx: while start_idx < byte_text.len() {
            let max_slice_len = std::cmp::min(self.max_token_len, byte_text.len() - start_idx);
            for slice_len in (1..=max_slice_len).rev() {
                let end_idx = start_idx + slice_len;
                let potential_token: Token = Arc::from(&byte_text[start_idx..end_idx]);
                if self.token_to_id.contains_key(&potential_token) {
                    tokens.push(potential_token);
                    start_idx += slice_len;
                    continue 'first_idx;
                }
            }

            return Err(Error::msg(format!(
                "No token found for slice: {}",
                String::from_utf8(byte_text[start_idx..].to_vec()).unwrap_or_else(|_| {
                    format!("Invalid UTF-8 sequence: {:?}", &byte_text[start_idx..])
                })
            )));
        }
        Ok(tokens)
    }

    /// Encode a list of strings into a list of token IDs
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<TokenId>>> {
        texts
            .par_iter()
            .map(|&text| {
                self.split_into_tokens(text)?
                    .iter()
                    .map(|token| Ok(*self.token_to_id.get(token).unwrap()))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Decode a list of token IDs into a string
    pub fn decode(&self, token_id_lists: &[&[TokenId]]) -> Vec<String> {
        token_id_lists
            .par_iter()
            .map(|token_ids| {
                token_ids
                    .iter()
                    .map(|token_id| {
                        String::from_utf8(self.id_to_token[*token_id as usize].to_vec()).unwrap()
                    })
                    .fold(String::new(), |mut acc, token| {
                        acc.push_str(&token);
                        acc
                    })
            })
            .collect()
    }
}

/// Train a tokenizer on a given corpus file and optionally save the tokenizer to a file.
/// Todo: This is currently slow since it walks through the corpus multiple times, we can
/// improve this by storing a linked-list of occurrences of each pair in the corpus, so
/// that we don't need to re-scan the corpus for each token merge.
///
/// Args:
/// - `corpus_file`: Path to the corpus file.
/// - `vocab_size`: Size of the vocabulary.
/// - `unk_token_id`: ID of the unknown token.
/// - `output_file`: Optional path to save the tokenizer.
/// Returns:
/// - A vector of tokens.
pub fn train_tokenizer(
    corpus_file: &Path,
    vocab_size: usize,
    special_tokens: &[&str],
    output_file_path: Option<&Path>,
) -> Vec<Token> {
    // Initialize vocabulary with byte tokens
    let mut vocab: Vec<Token> = (0..=u8::MAX).map(|b| Arc::from([b])).collect();

    let corpus_string = std::fs::read_to_string(corpus_file)
        .unwrap_or_else(|_| panic!("Failed to read corpus file: {:?}", corpus_file));

    // Corpus is represented as vector of byte token IDs
    let mut corpus: Vec<TokenId> = corpus_string.bytes().map(|b| b as TokenId).collect();

    let mut bigram_counts: HashMap<u64, usize> = HashMap::new();
    for w in corpus.windows(2) {
        *bigram_counts
            .entry(bigram_to_u64((w[0], w[1])))
            .or_insert(0) += 1;
    }
    let mut lazy_bigram_heap: BinaryHeap<(usize, u64)> = bigram_counts
        .iter()
        .map(|(&token_id_pair, &pair_count)| (pair_count, token_id_pair))
        .collect();

    while vocab.len() < vocab_size {
        let start = std::time::Instant::now();
        let (max_bigram_count, token_id_pair) = match lazy_bigram_heap.pop() {
            Some(x) => x,
            None => {
                println!("No more bigrams found. Stopping training.");
                break;
            }
        };
        let (token_id_1, token_id_2) = u64_to_bigram(token_id_pair);

        // Check if entry is valid
        if max_bigram_count != bigram_counts[&token_id_pair] {
            continue;
        }

        println!(
            "Current vocabulary size: {}. Target vocabulary size: {}",
            vocab.len(),
            vocab_size
        );

        let token_1 = vocab[token_id_1 as usize].clone();
        let token_2 = vocab[token_id_2 as usize].clone();

        let new_token: Token = Arc::from(
            token_1
                .iter()
                .chain(token_2.iter())
                .copied()
                .collect::<Token>(),
        );

        let new_token_string = String::from_utf8(new_token.to_vec())
            .unwrap_or_else(|_| format!("Failed to convert token to string: {:?}", new_token));
        println!(
            "Most frequent bigram: `{}` with count: {} (token_id_1: {}, token_id_2: {})",
            new_token_string, max_bigram_count, token_id_1, token_id_2
        );
        vocab.push(new_token.clone());
        let new_token_id = (vocab.len() - 1) as TokenId;

        let mut write_pos = 0;
        let mut read_pos = 0;
        while read_pos < corpus.len() - 1 {
            if corpus[read_pos] == token_id_1 && corpus[read_pos + 1] == token_id_2 {
                // update the bigram counts
                if read_pos > 0 {
                    // Needs to be the newly written token, e.g. if we're merging a and b,
                    // then "a b a b" would otherwise fail to increment (ab, ab) and instead
                    // increment (ab, a) and (b, ab)
                    let prev_token_id = corpus[write_pos - 1];

                    // Update the bigram counts for the new token
                    let new_entry = bigram_counts
                        .entry(bigram_to_u64((prev_token_id, new_token_id)))
                        .or_insert(0);
                    *new_entry += 1;
                    lazy_bigram_heap
                        .push((*new_entry, bigram_to_u64((prev_token_id, new_token_id))));
                    // Remove the old bigram count
                    let old_entry = bigram_counts
                        .entry(bigram_to_u64((prev_token_id, token_id_1)))
                        .or_insert(0);
                    if *old_entry == 0 {
                        panic!(
                            "Error: old_entry for ({}, {}) is 0",
                            prev_token_id, token_id_1
                        );
                    }
                    *old_entry -= 1;
                    lazy_bigram_heap.push((*old_entry, bigram_to_u64((prev_token_id, token_id_1))));
                }
                if read_pos + 2 < corpus.len() {
                    let next_token_id = corpus[read_pos + 2];
                    // Update the bigram counts for the new token
                    let new_entry = bigram_counts
                        .entry(bigram_to_u64((new_token_id, next_token_id)))
                        .or_insert(0);
                    *new_entry += 1;
                    lazy_bigram_heap
                        .push((*new_entry, bigram_to_u64((new_token_id, next_token_id))));
                    // Remove the old bigram count
                    let old_entry = bigram_counts
                        .entry(bigram_to_u64((token_id_2, next_token_id)))
                        .or_insert(0);
                    if *old_entry == 0 {
                        panic!(
                            "Error: old_entry for ({}, {}) is 0",
                            token_id_2, next_token_id
                        );
                    }
                    *old_entry -= 1;
                    lazy_bigram_heap.push((*old_entry, bigram_to_u64((token_id_2, next_token_id))));
                }

                corpus[write_pos] = new_token_id;
                read_pos += 2;
            } else {
                corpus[write_pos] = corpus[read_pos];
                read_pos += 1;
            }
            write_pos += 1;
        }
        if read_pos < corpus.len() {
            corpus[write_pos] = corpus[read_pos];
            write_pos += 1;
        }
        corpus.truncate(write_pos);
        bigram_counts.insert(bigram_to_u64((token_id_1, token_id_2)), 0);

        let num_tokens_to_add = vocab_size - vocab.len();
        let time_for_this_token = start.elapsed();
        if num_tokens_to_add > 0 {
            println!("Remaining tokens to add: {}\nTime taken for this token: {:?}\nEstimated time remaining: {:?}\n>>>>>>>>>>>>>>>>>>>>>>", num_tokens_to_add, time_for_this_token, time_for_this_token * num_tokens_to_add as u32);
        }
    }

    // Add special tokens to the vocabulary
    println!("Adding special tokens to the vocabulary...");
    let special_tokens_vec: Vec<Token> = special_tokens
        .iter()
        .map(|&s| Arc::from(s.as_bytes()))
        .collect();
    vocab.extend(special_tokens_vec.clone());
    println!(
        "Added {} special tokens to the vocabulary: {:?}",
        special_tokens.len(),
        special_tokens
    );

    // Save as a json array of base64 strings if output_file is provided
    if let Some(output_file_path) = output_file_path {
        println!("Saving tokenizer to file: {:?}", output_file_path);
        save_vocab_json(&vocab, output_file_path)
            .unwrap_or_else(|_| panic!("Failed to save tokenizer to file: {:?}", output_file_path));
        save_vocab_json_string(&vocab, &output_file_path.with_extension("txt")).unwrap_or_else(
            |_| {
                panic!(
                    "Failed to save text tokenizer to file: {:?}",
                    output_file_path
                )
            },
        );
        println!("Tokenizer saved successfully.");
    }
    println!("Tokenizer training completed.");
    vocab
}

fn save_vocab_json(tokens: &[Token], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let vocab_b64: Vec<String> = tokens
        .iter()
        .map(|token| general_purpose::STANDARD.encode(token))
        .collect();
    serde_json::to_writer_pretty(writer, &vocab_b64)?;
    Ok(())
}

fn save_vocab_json_string(tokens: &[Token], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let vocab_string: Vec<String> = tokens
        .iter()
        .map(|token| {
            String::from_utf8(token.to_vec()).unwrap_or_else(|_| "<failed_utf8_decode>".to_string())
        })
        .collect();
    serde_json::to_writer_pretty(writer, &vocab_string)?;
    Ok(())
}

fn load_vocab_json(path: &Path) -> Result<Vec<Token>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let vocab_b64: Vec<String> = serde_json::from_reader(reader)?;

    let tokens: Result<Vec<Token>> = vocab_b64
        .into_iter()
        .map(|s| {
            let bytes = general_purpose::STANDARD
                .decode(&s)
                .map_err(anyhow::Error::from)?;
            Ok(Arc::<[u8]>::from(bytes))
        })
        .collect();

    tokens
}
