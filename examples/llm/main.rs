use ice_nine::{
    cross_entropy_labels, cross_entropy_llm_labels, get_last_col, sample_token, AdamW, LayerNorm, Linear, Model, Optimizer, PaddingStrategy, Sequential, Tensor, Tokenizer
};

pub fn main() {
    let vocab_file_path =
        std::path::Path::new("/Users/benchen/workspace/ice-nine/data/wikitext103_tok.json");

    // Define the model
    let tokenizer = Tokenizer::load_from_file(vocab_file_path).expect("Failed to load tokenizer");
    let vocab_size = tokenizer.vocab_size();
    let d_model = 30;
    let d_ff = 20;
    let d_head = 20;
    let embed_layer: Box<dyn Model<f32>> = Box::new(Linear::<f32>::random(vocab_size, d_model));
    let layer_norm1: Box<dyn Model<f32>> =
        Box::new(LayerNorm::<f32>::new(d_model, 1e-8));
    let linear_layer = Box::new(Linear::<f32>::random(d_model, d_model));
    let unembed_layer = Box::new(Linear::<f32>::random(d_model, vocab_size));
    let gelu1 = Box::new(ice_nine::Gelu::new());
    let gelu2 = Box::new(ice_nine::Gelu::new());
    let layer_norm2 = Box::new(ice_nine::LayerNorm::new(d_model, 1e-8));


    let transformer_block: Box<dyn Model<f32>> =
        Box::new(ice_nine::TransformerBlock::random(d_model, d_head, d_ff));

    // let layers = vec![embed_layer, transformer_block, unembed_layer];
    // let layers = vec![embed_layer, layer_norm1, gelu1, layer_norm2, linear_layer, gelu2, unembed_layer];
    let layers = vec![
        embed_layer,
        unembed_layer,
    ];

    let my_llm = Sequential::new(layers);
    let text = "hello world";
    println!("Original text: {:?}", text);
    let text_tokens = &tokenizer
        .encode(&[text], false, PaddingStrategy::None)
        .unwrap()[0];
    let text_tokens_for_loss = &text_tokens
        .into_iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>()[1..]
        .to_vec();
    println!("Tokenized text for loss: {:?}", text_tokens_for_loss);

    let text_for_input = "hello ";
    let input: Tensor<f32> = tokenizer
        .encode_one_hot(text_for_input, false, PaddingStrategy::None, true)
        .unwrap();
    println!("Input tensor shape: {:?}", input.shape());

    let mut optimizer = AdamW::new(0.05, my_llm.parameters(), 0.00, 0.9, 0.9, 1e-8);

    while optimizer.step_num() < 1000 {
        println!(">>>>>>>>>>>>>");
        println!("Step: {:?}", optimizer.step_num());
        let output = &my_llm.forward(&input);
        println!("Output tensor shape: {:?}", output.shape());

        eprintln!("Text tokens for loss: {:?}", text_tokens_for_loss);
        let mut loss = cross_entropy_labels(output, &text_tokens_for_loss);
        loss.backward().unwrap();
        // println!("Model: {:?}", model);
        println!("Loss: {:?}", loss.data()[0]);

        let text_tokens = &tokenizer
            .encode(&[text], false, PaddingStrategy::LongestInBatch)
            .unwrap()[0];
        let last_token_id = *text_tokens.last().unwrap();
        let text_tokens = &text_tokens[..text_tokens.len() - 1];

        let decoded_input = tokenizer.decode(&[text_tokens.to_vec()]);
        println!("Decoded input: {:?}", decoded_input);

        let text_tokens: Vec<_> = text_tokens.into_iter().map(|&x| x as usize).collect();
        println!("Tokenized text: {:?}", text_tokens);

        let input: Tensor<f32> = tokenizer
            .encode_one_hot(text_for_input, false, PaddingStrategy::None, true)
            .unwrap();
        let output_for_sampling = &my_llm.forward(&input);
        println!("Output tensor shape: {:?}", output_for_sampling.shape());

        let output_logits = &get_last_col(output_for_sampling);
        println!("Output logits shape: {:?}", output_logits.shape());
        let (output_sample, probs) = sample_token(output_logits, 1.0);
        let output_text = tokenizer.decode(&[vec![output_sample]]);
        let true_text = tokenizer.decode(&[vec![last_token_id]]);
        println!(
            "Sampled token: {:?}, prob: {:?}, id {:?}",
            output_text, probs[output_sample as usize], output_sample
        );
        println!(
            "True token: {:?}, prob: {:?}, id {:?}",
            true_text, probs[last_token_id as usize], last_token_id
        );
        let output_sample_index = 2 * output_sample as usize;
        println!(
            "Output logits for sample token: {} {} {}",
            output_for_sampling.data()[output_sample_index],
            output_for_sampling.data()[output_sample_index + 1],
            output_for_sampling.data()[output_sample_index + 2]
        );
        let true_text_index = 2 * last_token_id as usize;
        println!(
            "Output logits for true token: {} {} {}",
            output_for_sampling.data()[true_text_index],
            output_for_sampling.data()[true_text_index + 1],
            output_for_sampling.data()[true_text_index + 2]
        );
        println!(
            "Output logits for first token: {} {} {}",
            output_for_sampling.data()[0],
            output_for_sampling.data()[1],
            output_for_sampling.data()[2]
        );
        eprintln!(
            "'lo' logit: {:?} {:?}",
            output.data()[28734 * 2],
            output.data()[28734 * 2 + 1],
        );
        eprintln!(
            "'lo' logit grads: {:?} {:?}",
            output.grad().unwrap().data[28734 * 2],
            output.grad().unwrap().data[28734 * 2 + 1],
        );
        eprintln!(
            "'world' logit: {:?} {:?}",
            output.data()[4898 * 2],
            output.data()[4898 * 2 + 1],
        );
        eprintln!(
            "'world' logit grads: {:?} {:?}",
            output.grad().unwrap().data[4898 * 2],
            output.grad().unwrap().data[4898 * 2 + 1],
        );

        optimizer.step();
        optimizer.zero_grad();
    }

    println!("!!!!!!Done training!!!!!!");
}
