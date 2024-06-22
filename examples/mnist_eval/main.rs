use anyhow::Error;
use ice_nine::data::Dataset;
use ice_nine::layer::{LeakyRelu, Linear};
use ice_nine::loss::{logits_to_probs, CrossEntropy};
use ice_nine::{Loss, Network};
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
    leaky_slope: f64,
    num_relu_layers: usize,
    test_data_path: String,
    load_weights_path: String,
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
    let config_content = std::fs::read_to_string(Path::new("mnist_eval_config.toml"))?;
    let config: Config = toml::from_str(&config_content)?;
    println!("Starting run: {}", config.run_name);

    // Initialize network
    let mut network = Network::new();

    // Relu layers
    let random_weights =
        Array2::from_shape_simple_fn((config.relu_layer_dim, config.input_dim), || 0.0);
    let layer = LeakyRelu::new_layer(random_weights, config.leaky_slope);
    network.push(layer)?;
    for _i in 1..config.num_relu_layers {
        let random_weights =
            Array2::from_shape_simple_fn((config.relu_layer_dim, config.relu_layer_dim), || 0.0);
        let layer = LeakyRelu::new_layer(random_weights, config.leaky_slope);
        network.push(layer)?;
    }

    // Linear layer to project down to 10 dims
    let random_weights =
        Array2::from_shape_simple_fn((config.output_dim, config.relu_layer_dim), || 0.0);
    let layer = Linear::new_layer(random_weights);
    network.push(layer)?;

    // Load weights
    println!("Loading weights from {}", config.load_weights_path);
    network.load_weights(Path::new(&config.load_weights_path))?;

    let ce_loss = Box::new(CrossEntropy {
        temperature: config.temperature,
    });

    let (inputs, targets) = load_mnist_data(Path::new(&config.test_data_path))?;
    let mut dataset = Dataset::new(&inputs, &targets);

    let mut num_correct = 0.0;
    for step in 1.. {
        let (input, target) = if let Some(data_pair) = dataset.next() {
            data_pair
        } else {
            println!("Finished eval on all data!");
            break;
        };
        println!("Step: {}", step);
        let output = network.f(input.clone());
        let probs = logits_to_probs(&output);
        num_correct += probs[*target];
        let running_accuracy = num_correct / step as f64;
        println!("Running accuracy: {}", running_accuracy);
        let loss = ce_loss.l(&output, target);
        println!("Loss: {}", loss);
        println!("--------------");
    }

    Ok(())
}
