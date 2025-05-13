use anyhow::{Error, Result};
use base64::{engine::general_purpose, Engine};
use rand::prelude::Distribution;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use crate::{DataType, Tensor};

pub type Token = Arc<[u8]>;

pub type TokenId = u32;

type Bigram = (TokenId, TokenId);

#[derive(Debug, Clone, Copy)]
pub enum PaddingStrategy {
    None,
    MaxLength(usize),
    LongestInBatch,
}

fn bigram_to_u64(bigram: Bigram) -> u64 {
    let (a, b) = bigram;
    ((a as u64) << 32) | (b as u64)
}

fn u64_to_bigram(bigram: u64) -> Bigram {
    let a = (bigram >> 32) as TokenId;
    let b = (bigram & 0xFFFFFFFF) as TokenId;
    (a, b)
}

const UNK_BYTES: &[u8] = "<unk>".as_bytes();
const BOS_BYTES: &[u8] = "<s>".as_bytes();
const EOS_BYTES: &[u8] = "</s>".as_bytes();
const PAD_BYTES: &[u8] = "<pad>".as_bytes();

type TokenAndId = (Token, TokenId);

pub struct Tokenizer {
    pub id_to_token: Arc<[Token]>,
    pub token_to_id: HashMap<Token, TokenId>,
    pub max_token_len: usize,
    pub unk_token_and_id: Option<TokenAndId>,
    pub bos_token_and_id: Option<TokenAndId>,
    pub eos_token_and_id: Option<TokenAndId>,
    pub pad_token_and_id: Option<TokenAndId>,
}

impl Tokenizer {
    pub fn new(tokens: Arc<[Token]>) -> Self {
        let mut max_token_len = 0;
        let token_to_id: HashMap<_, _> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| {
                max_token_len = std::cmp::max(max_token_len, token.len());
                (token.clone(), id as TokenId)
            })
            .collect();

        // Add special tokens
        let unk_token = &Arc::from(UNK_BYTES);
        let unk_token_and_id = token_to_id
            .get(unk_token)
            .map(|&id| (unk_token.clone(), id));
        let bos_token = &Arc::from(BOS_BYTES);
        let bos_token_and_id = token_to_id
            .get(bos_token)
            .map(|&id| (bos_token.clone(), id));
        let eos_token = &Arc::from(EOS_BYTES);
        let eos_token_and_id = token_to_id
            .get(eos_token)
            .map(|&id| (eos_token.clone(), id));
        let pad_token = &Arc::from(PAD_BYTES);
        let pad_token_and_id = token_to_id
            .get(pad_token)
            .map(|&id| (pad_token.clone(), id));

        Self {
            id_to_token: tokens,
            token_to_id,
            max_token_len,
            unk_token_and_id,
            bos_token_and_id,
            eos_token_and_id,
            pad_token_and_id,
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

    pub fn split_into_tokens(
        &self,
        texts: &[&str],
        add_special_characters: bool,
        padding_strategy: PaddingStrategy,
    ) -> Result<Vec<Vec<Token>>> {
        let mut token_lists = texts
            .into_par_iter()
            .map(|text| {
                let byte_text: Arc<_> = text.bytes().collect();

                let mut tokens = vec![];
                if add_special_characters {
                    if let Some((_, bos_tok_id)) = &self.bos_token_and_id {
                        tokens.push(self.id_to_token[*bos_tok_id as usize].clone());
                    }
                }
                let mut start_idx = 0;
                'first_idx: while start_idx < byte_text.len() {
                    let max_slice_len =
                        std::cmp::min(self.max_token_len, byte_text.len() - start_idx);
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

                if add_special_characters {
                    if let Some((_, eos_tok_id)) = &self.eos_token_and_id {
                        tokens.push(self.id_to_token[*eos_tok_id as usize].clone());
                    }
                }

                Ok(tokens)
            })
            .collect::<Result<Vec<_>>>()?;

        let pad_length = match padding_strategy {
            PaddingStrategy::MaxLength(max_length) => Some(max_length),
            PaddingStrategy::LongestInBatch => {
                let longest = token_lists
                    .iter()
                    .map(|tokens| tokens.len())
                    .max()
                    .unwrap_or(0);
                Some(longest)
            }
            PaddingStrategy::None => None,
        };
        if add_special_characters {
            match pad_length {
                Some(length) => {
                    if let Some((_, pad_tok_id)) = &self.pad_token_and_id {
                        token_lists.par_iter_mut().for_each(|tokens| {
                            let pad_len = length - tokens.len();
                            let pad_token = self.id_to_token[*pad_tok_id as usize].clone();
                            tokens.extend(std::iter::repeat(pad_token).take(pad_len));
                        });
                    } else {
                        return Err(Error::msg(format!(
                            "Pad token should be set for PaddingStrategy {:?}",
                            padding_strategy
                        )));
                    }
                }
                None => (),
            }
        }
        Ok(token_lists)
    }

    /// Return the size of the vocabulary
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode a list of strings into a list of token IDs
    pub fn encode(
        &self,
        texts: &[&str],
        add_special_characters: bool,
        padding_strategy: PaddingStrategy,
    ) -> Result<Vec<Vec<TokenId>>> {
        let tokens_lists =
            self.split_into_tokens(texts, add_special_characters, padding_strategy)?;

        let token_id_lists = tokens_lists
            .into_par_iter()
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| self.token_to_id[&token] as TokenId)
                    .collect::<Vec<TokenId>>()
            })
            .collect::<Vec<_>>();
        Ok(token_id_lists)
    }

    pub fn encode_one_hot<A: DataType>(
        &self,
        text: &str,
        add_special_characters: bool,
        padding_strategy: PaddingStrategy,
        require_grad: bool,
    ) -> Result<Tensor<A>> {
        let vocab_size = self.vocab_size();
        let token_id_lists = self.encode(&[text], add_special_characters, padding_strategy)?;
        let token_id_list = token_id_lists.last().unwrap();
        let seq_len = token_id_list.len();
        let mut one_hot_data = vec![A::zero(); vocab_size * seq_len];
        for (n, &token_id) in token_id_list.iter().enumerate() {
            one_hot_data[n * seq_len + token_id as usize] = A::one();
        }

        let one_hot_tensor = Tensor::new(
            &[vocab_size, seq_len],
            Arc::from(one_hot_data),
            require_grad,
        );
        Ok(one_hot_tensor)
    }

    /// Decode a list of token IDs into a string
    pub fn decode(&self, token_id_lists: &[Vec<TokenId>]) -> Vec<String> {
        token_id_lists
            .par_iter()
            .map(|token_ids| {
                let bytes: Vec<u8> = token_ids
                    .into_iter()
                    .flat_map(|token_id| self.id_to_token[*token_id as usize].to_vec())
                    .collect();

                String::from_utf8(bytes).unwrap_or_else(|_| format!("<Invalid UTF-8 sequence>"))
            })
            .collect()
    }

    ///Decode from one-hot tensor
    pub fn decode_one_hot(&self, one_hot_tensor: &Tensor<f32>) -> Vec<String> {
        let token_id_lists = one_hot_tensor
            .data()
            .par_iter()
            .enumerate()
            .map(|(i, &value)| {
                if value > 0.5 {
                    Some(i as TokenId)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let token_ids = token_id_lists
            .into_iter()
            .filter_map(|x| x)
            .collect::<Vec<TokenId>>();

        self.decode(&[token_ids])
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

        let new_token: Token = token_1
            .iter()
            .chain(token_2.iter())
            .copied()
            .collect::<Token>();

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

pub fn get_last_col<A: DataType>(x: &Tensor<A>) -> Tensor<A> {
    let num_cols = x.shape()[1];
    let last_col = x
        .data()
        .iter()
        .step_by(num_cols)
        .cloned()
        .collect::<Vec<_>>();
    Tensor::new(&[x.shape()[0], 1], Arc::from(last_col), false)
}

pub fn sample_token(logits: &Tensor<f32>, temperature: f32) -> (TokenId, Vec<f32>) {
    let logits_array = &logits.array();
    let logits_array = logits_array / temperature;
    let logits_tensor = Tensor::new(&[logits_array.dim[0], 1], logits_array.data, false);
    let probs_tensor = logits_tensor.softmax_col();
    let probs_tensor_data: Arc<Vec<_>> = probs_tensor.data();

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::WeightedIndex::new(probs_tensor_data.iter()).unwrap();

    let sample = dist.sample(&mut rng);
    (sample as TokenId, probs_tensor_data.to_vec())
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
