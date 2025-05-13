use hillock::StopWatch;
use ice_nine::{
    cross_entropy_labels, display_loss, AdamW, Linear, Model, Optimizer, PaddingStrategy,
    Sequential, Tensor, Tokenizer,
};

const MAX_TRAINING_STEPS: usize = 1000;
const WORDS_PER_CHUNK: usize = 3;
const LOSS_DISPLAY_INTERVAL: usize = 5;

pub fn main() {
    // Load the tokenizer
    let vocab_file_path =
        std::path::Path::new("/Users/benchen/workspace/ice-nine/data/wikitext103_tok.json");
    let tokenizer = Tokenizer::load_from_file(vocab_file_path).expect("Failed to load tokenizer");

    // Define the model
    let vocab_size = tokenizer.vocab_size();
    let d_model = 20;
    let d_ff = 20;
    let d_head = 20;
    let embed_layer: Box<dyn Model<f32>> = Box::new(Linear::<f32>::random(vocab_size, d_model));
    let unembed_layer = Box::new(Linear::<f32>::random(d_model, vocab_size));
    let transformer_block: Box<dyn Model<f32>> =
        Box::new(ice_nine::TransformerBlock::random(d_model, d_head, d_ff));
    let layers = vec![embed_layer, transformer_block, unembed_layer];
    let my_llm = Sequential::new(layers);

    let num_params = my_llm.parameters().iter().map(|x| x.numel()).sum::<usize>();
    println!("Number of parameters: {:?}", num_params);

    // Define the optimizer
    let mut optimizer = AdamW::new(0.001, my_llm.parameters(), 0.01, 0.9, 0.95, 1e-8);

    // Load the training data
    let training_data_path =
        std::path::Path::new("/Users/benchen/workspace/ice-nine/data/wikitext103.txt");
    let training_data =
        std::fs::read_to_string(training_data_path).expect("Failed to read training data");

    // Training loop
    let mut losses = vec![];
    for chunk in training_data
        .split_whitespace()
        .collect::<Vec<_>>()
        .chunks(WORDS_PER_CHUNK)
    {
        let mut stopwatch = StopWatch::new(true, false);
        stopwatch.reset();
        if optimizer.step_num() >= MAX_TRAINING_STEPS {
            break;
        }
        println!(">>>>>>>>>>>>>");
        println!("Step: {:?}", optimizer.step_num());

        // let text = chunk.join(" ");
        let text = "hello world";
        eprintln!("Full text: {:?}", text);

        let text_tokens = &tokenizer
            .encode(&[&text], false, PaddingStrategy::None)
            .unwrap()[0];
        if text_tokens.len() < 2 {
            continue;
        }
        let text_tokens_for_input = &text_tokens[..text_tokens.len() - 1];
        // eprintln!("Tokens for input: {:?}", text_tokens_for_input);
        let text_tokens_for_output = &text_tokens[1..];
        // eprintln!("Tokens for output: {:?}", text_tokens_for_output);

        let text_for_input = tokenizer
            .decode(&[text_tokens_for_input.to_vec()])
            .first()
            .unwrap()
            .clone();
        eprintln!("Text for input: {:?}", text_for_input);
        let text_for_output = tokenizer
            .decode(&[text_tokens_for_output.to_vec()])
            .first()
            .unwrap()
            .clone();
        eprintln!("Text for output: {:?}", text_for_output);
        let input: Tensor<f32> = tokenizer
            .encode_one_hot(&text_for_input, false, PaddingStrategy::None, true)
            .unwrap();
        stopwatch.tick("tokenize");

        println!("Input tensor:\n{:?}", input.shape());
        let output = &my_llm.forward(&input);
        println!("Output tensor:\n{:?}", output.shape());
        stopwatch.tick("forward");

        let mut loss = cross_entropy_labels(output, text_tokens_for_output);
        println!("Loss: {:?}", loss.data()[0]);
        stopwatch.tick("loss");

        loss.backward().unwrap();
        losses.push(loss.data()[0]);
        stopwatch.tick("backward");

        optimizer.step();
        stopwatch.tick("step");
        optimizer.zero_grad();
        stopwatch.tick("zero_grad");
        stopwatch.breakdown(1, "step");

        if optimizer.step_num() % LOSS_DISPLAY_INTERVAL == 0 {
            println!("Loss: {:?}", loss.data()[0]);
            display_loss(&losses);
        }
    }

    println!("!!!!!!Done training!!!!!!");
    display_loss(&losses);
}
