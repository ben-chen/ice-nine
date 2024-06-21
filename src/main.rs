use ice_nine::grad::{Network, Optimizer};
use ice_nine::layers::Relu;
use ice_nine::losses::LeastSquaresLoss;

fn main() -> Result<(), ndarray::ShapeError> {
    let mut network = Network::new();
    for _i in 0..5 {
        let random_weights =
            ndarray::Array2::from_shape_simple_fn((5, 5), rand::random::<f64>);
            // 0.5*ndarray::Array2::eye(5);
        let layer = Relu::new_layer(random_weights);
        network.push(layer)?;
    }
    let random_weights = ndarray::Array2::from_shape_simple_fn((3, 5), || 0.5);
    dbg!(&random_weights);
    let layer = Relu::new_layer(random_weights);
    network.push(layer)?;
    let least_squares_loss = Box::new(LeastSquaresLoss {});

    let random_weights = ndarray::Array2::from_shape_simple_fn((5, 3), || 0.5);
    dbg!(&random_weights);
    let layer = Relu::new_layer(random_weights);
    network.push(layer)?;

    let mut optimizer = Optimizer {
        loss: least_squares_loss,
        network,
    };

    let random_input = ndarray::Array1::from_shape_simple_fn(5, rand::random::<f64>);
    println!("Random input: {:.2}", random_input);
    let network_output = optimizer.apply(&random_input, &(2.0 * random_input.clone()));
    println!("Network output: {:.2}", network_output.0);
    println!("Loss: {:.2}", network_output.1);

    Ok(())
}
