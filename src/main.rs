use anyhow::Error;
use ice_nine::data::Dataset;
use ice_nine::grad::{Network, Optimizer};
use ice_nine::layers::{LeakyRelu, Linear};
use ice_nine::losses::{logits_to_probs, CrossEntropyLoss};
use ndarray::{Array1, Array2};
use std::io::BufRead;
use std::path::Path;

const RUN_NAME: &str = "MNIST example";

const RELU_LAYER_DIM: usize = 784;
const INPUT_DIM: usize = 784;
const OUTPUT_DIM: usize = 10;
const TEMPERATURE: f64 = 50.0;

const STEPS_TO_TRAIN: usize = 200;
const MICROBATCH_SIZE: usize = 20;
const LEARNING_RATE: f64 = 1e-2;
const MAX_GRADIENT: f64 = 1e4;
const LEAKY_SLOPE: f64 = 1e-2;
const NUM_RELU_LAYERS: usize = 20;

const TRAIN_DATA_PATH: &str = "/Users/benchen/workspace/ice-nine/data/mnist_train.csv";
// const TEST_DATA_PATH: &str = "";
const SAVE_WEIGHTS_PATH: &str = "/Users/benchen/workspace/ice-nine/weights/mnist.bin";
const LOAD_WEIGHTS_PATH: Option<&str> = None;

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
    println!("Starting run: {}", RUN_NAME);

    // Initialize network
    let mut network = Network::new();

    // Relu layers
    let random_weights = Array2::from_shape_simple_fn((RELU_LAYER_DIM, INPUT_DIM), random_weight);
    let layer = LeakyRelu::new_layer(random_weights, LEAKY_SLOPE);
    network.push(layer)?;
    for _i in 1..NUM_RELU_LAYERS {
        let random_weights =
            Array2::from_shape_simple_fn((RELU_LAYER_DIM, RELU_LAYER_DIM), random_weight);
        let layer = LeakyRelu::new_layer(random_weights, LEAKY_SLOPE);
        network.push(layer)?;
    }

    // Linear layer to project down to 10 dims
    let random_weights = Array2::from_shape_simple_fn((OUTPUT_DIM, RELU_LAYER_DIM), random_weight);
    let layer = Linear::new_layer(random_weights);
    network.push(layer)?;

    // Load weights
    if let Some(load_weights_path) = LOAD_WEIGHTS_PATH {
        println!("Loading weights from {}", load_weights_path);
        network.load_weights(Path::new(load_weights_path))?;
    }

    let ce_loss = Box::new(CrossEntropyLoss {
        temperature: TEMPERATURE,
    });
    let mut optimizer = Optimizer {
        loss: ce_loss,
        network,
        max_gradient: Some(MAX_GRADIENT),
    };

    let (inputs, targets) = load_mnist_data(Path::new(TRAIN_DATA_PATH))?;
    let mut dataset = Dataset::new(&inputs, &targets);

    for step in 0..STEPS_TO_TRAIN {
        let mut num_correct = 0.0;
        let mut microbatch_loss = 0.0;
        dbg!(step);
        for _microbatch_num in 0..MICROBATCH_SIZE {
            let (input, target) = if let Some(data_pair) = dataset.next() {
                data_pair
            } else {
                break;
            };

            let (output, loss) = optimizer.apply(input, target);
            // dbg!(microbatch_num);
            // dbg!(target);
            let probs = logits_to_probs(&output);
            // dbg!(probs[*target]);
            num_correct += probs[*target];
            microbatch_loss += loss;
            // dbg!(loss);
            // dbg!(&probs);
            // dbg!(&output);
        }
        let microbatch_accuracy = num_correct / MICROBATCH_SIZE as f64;
        microbatch_loss /= MICROBATCH_SIZE as f64;
        dbg!(microbatch_accuracy);
        dbg!(microbatch_loss);
        println!("--------------");

        optimizer.step(LEARNING_RATE / MICROBATCH_SIZE as f64);
    }

    println!("Saving weights to {}", SAVE_WEIGHTS_PATH);
    optimizer.network.save_weights(Path::new(SAVE_WEIGHTS_PATH))
}
