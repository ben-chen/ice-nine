use anyhow::Error;
use ice_nine::data::Dataset;
use ice_nine::layer::{LeakyRelu, Linear};
use ice_nine::loss::{logits_to_probs, CrossEntropy};
use ice_nine::{Network, Optimizer};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::io::BufRead;
use std::path::Path;

#[derive(Deserialize)]
struct Config {
    run_name: String,
    relu_layer_dim: usize,
    input_dim: usize,
    output_dim: usize,
    temperature: f64,
    steps_to_train: usize,
    microbatch_size: usize,
    learning_rate: f64,
    max_gradient: f64,
    leaky_slope: f64,
    num_relu_layers: usize,
    train_data_path: String,
    test_data_path: String,
    save_weights_path: String,
    load_weights_path: Option<String>,
    steps_per_test: usize,
    num_test_examples: usize,
}

fn random_weight() -> f64 {
    (rand::random::<f64>() - 0.5) * 0.15
}

fn read_csv(path: &Path) -> Result<Vec<Vec<usize>>, Error> {
    let mut result = Vec::new();
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let numbers: Vec<usize> = line
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<Vec<_>, _>>()?;
        result.push(numbers);
    }

    Ok(result)
}

fn load_mnist_data(data_path: &Path) -> Result<(Vec<Array1<f64>>, Vec<usize>), Error> {
    let raw_nums = read_csv(data_path)?;
    let inputs = raw_nums
        .iter()
        .map(|nums| Array1::from_vec(nums[1..].iter().map(|n| *n as f64).collect()))
        .collect();
    let targets = raw_nums.iter().map(|nums| nums[0]).collect();

    Ok((inputs, targets))
}

fn main() -> Result<(), Error> {
    let config_content = std::fs::read_to_string("mnist_train_config.toml")?;
    let config: Config = toml::from_str(&config_content)?;
    println!("Starting run: {}", config.run_name);

    // Initialize network
    let mut network = Network::new();

    // Relu layers
    let random_weights =
        Array2::from_shape_simple_fn((config.relu_layer_dim, config.input_dim), random_weight);
    let layer = LeakyRelu::new_layer(random_weights, config.leaky_slope);
    network.push(layer)?;
    for _i in 1..config.num_relu_layers {
        let random_weights = Array2::from_shape_simple_fn(
            (config.relu_layer_dim, config.relu_layer_dim),
            random_weight,
        );
        let layer = LeakyRelu::new_layer(random_weights, config.leaky_slope);
        network.push(layer)?;
    }

    // Linear layer to project down to 10 dims
    let random_weights =
        Array2::from_shape_simple_fn((config.output_dim, config.relu_layer_dim), random_weight);
    let layer = Linear::new_layer(random_weights);
    network.push(layer)?;

    // Load weights
    if let Some(load_weights_path) = config.load_weights_path {
        println!("Loading weights from {}", load_weights_path);
        network.load_weights(Path::new(&load_weights_path))?;
    }

    let ce_loss = Box::new(CrossEntropy {
        temperature: config.temperature,
    });
    let mut optimizer = Optimizer {
        loss: ce_loss,
        network,
        max_gradient: Some(config.max_gradient),
    };

    let (train_inputs, train_targets) = load_mnist_data(Path::new(&config.train_data_path))?;
    let mut train_dataset = Dataset::new(&train_inputs, &train_targets)?;
    let (test_inputs, test_targets) = load_mnist_data(Path::new(&config.test_data_path))?;
    let mut test_dataset = Dataset::new(&test_inputs, &test_targets)?;

    'training_loop: for step in 1..config.steps_to_train + 1 {
        let mut num_correct = 0.0;
        let mut microbatch_loss = 0.0;
        println!("Step: {}", step);
        for _microbatch_num in 0..config.microbatch_size {
            let (input, target) = if let Some(data_pair) = train_dataset.next() {
                data_pair
            } else {
                println!("Finished training on all data!");
                break 'training_loop;
            };

            let (output, loss) = optimizer.apply(input, target);
            let probs = logits_to_probs(&output);
            num_correct += probs[*target];
            microbatch_loss += loss;
        }
        let microbatch_accuracy = num_correct / config.microbatch_size as f64;
        microbatch_loss /= config.microbatch_size as f64;
        println!("Microbatch accuracy: {}", microbatch_accuracy);
        println!("Microbatch loss: {}", microbatch_loss);
        println!("--------------");

        optimizer.step(config.learning_rate / config.microbatch_size as f64);

        if step % config.steps_per_test == 0 {
            test(&optimizer.network, &mut test_dataset, config.num_test_examples);
        }
    }

    println!("Saving weights to {}", config.save_weights_path);
    optimizer
        .network
        .save_weights(Path::new(&config.save_weights_path))
}

fn test(network: &Network, test_dataset: &mut Dataset<usize>, num_test_examples: usize) {
    println!("Testing on {} examples...", num_test_examples);
    if num_test_examples == 0 {
        return;
    }
    let mut num_correct = 0;
    for _i in 0..num_test_examples {
        let (input, target) = if let Some(data_pair) = test_dataset.next() {
            data_pair
        } else {
            panic!("")
        };
        let output = network.f(input.clone());
        let probs = logits_to_probs(&output);
        let mut pred_digit = 0;
        let mut max_prob = probs[0];
        for i in 1..probs.len() {
            if probs[i] > max_prob {
                max_prob = probs[i];
                pred_digit = i;
            }
        }
        num_correct += (*target == pred_digit) as usize;
    }
    let accuracy = num_correct as f64 / num_test_examples as f64;
    println!("Accuracy: {}", accuracy);
}
