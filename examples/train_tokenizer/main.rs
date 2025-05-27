use ice_nine::train_tokenizer;

pub fn main() {
    let corpus_file =
        std::path::Path::new("/Users/benchen/workspace/ice-nine/data/wikitext103.txt");
    let vocab_size = 50_000;
    let unk_string = "<unk>";
    let bos_string = "<s>";
    let eos_string = "</s>";
    let pad_string = "<pad>";

    let special_tokens = &[unk_string, bos_string, eos_string, pad_string];
    let output_file = Some(std::path::Path::new(
        "/Users/benchen/workspace/ice-nine/data/wikitext103_tok.json",
    ));
    println!("Training tokenizer on corpus file: {:?}", corpus_file);

    let tokens = train_tokenizer(&corpus_file, vocab_size, special_tokens, output_file);

    println!("Vocabulary size: {}", tokens.len());
}
