use anyhow::Error;
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array1, Array2, ErrorKind, ShapeError};
use std::io::{Read, Write};
use std::path::Path;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    /// Run inference without calculating gradients
    pub fn f(&self, v: Array1<f64>) -> Array1<f64> {
        self.layers.iter().fold(v, |prev_v, layer| layer.f(&prev_v))
    }

    /// Add a new layer at the end
    pub fn push(&mut self, layer: Layer) -> Result<(), ShapeError> {
        if let Some(last_layer) = self.layers.last() {
            if last_layer.dims().0 != layer.dims().1 {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Create a new network with no layers
    pub fn new() -> Self {
        Network { layers: Vec::new() }
    }

    /// Update weights with stored gradients
    pub fn update(&mut self, learning_rate: f64) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.update(learning_rate));
    }

    /// Reset gradients to 0
    pub fn zero_grad(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.zero_grad());
    }

    /// Save the weights as a binary file
    pub fn save_weights(&self, output_path: &Path) -> Result<(), Error> {
        let output_dir = match output_path.parent() {
            Some(output_dir) => output_dir,
            None => return Err(Error::msg("Failed to get output_path directory")),
        };
        std::fs::create_dir_all(output_dir)?;
        let mut output_file = std::fs::File::create(output_path)?;
        let bytes: Vec<u8> = self
            .layers
            .iter()
            .flat_map(|layer| layer.weights.clone().into_raw_vec())
            .flat_map(|x| x.to_le_bytes())
            .collect();
        output_file.write_all(&bytes)?;
        Ok(())
    }

    /// Load the weights from a binary file
    /// Returns ErrorKind::UnexpectedEof if there are not enough weights
    pub fn load_weights(&mut self, input_path: &Path) -> Result<(), Error> {
        let mut input_file = std::fs::File::open(input_path)?;
        let mut bytes = Vec::new();
        input_file.read_to_end(&mut bytes)?;

        let mut offset = 0;
        for layer in &mut self.layers {
            let num_elements = layer.weights.len();
            let mut raw_vec = Vec::with_capacity(num_elements);
            for _i in 0..num_elements {
                let value = (&bytes[offset..offset + 8]).read_f64::<LittleEndian>()?;
                raw_vec.push(value);
                offset += 8; // 8 = num bytes in f64
            }
            layer.weights = Array2::from_shape_vec(layer.weights.raw_dim(), raw_vec)?;
        }
        Ok(())
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Layer {
    pub activation: Box<dyn HasGrad<f64>>,
    pub weights: Array2<f64>,
    pub gradients: Array2<f64>,
}

impl Layer {
    /// Apply the layer, updating gradients
    pub fn f(&self, v: &Array1<f64>) -> Array1<f64> {
        let linear = self.weights.dot(v);
        linear.mapv_into(|x| self.activation.f(x))
    }

    /// Get weight matrix dims as a tuple
    pub fn dims(&self) -> (usize, usize) {
        self.weights.dim()
    }

    /// Update layer weights in-place
    pub fn update(&mut self, learning_rate: f64) {
        self.weights -= &(learning_rate * &self.gradients);
    }

    /// Set gradients to 0
    pub fn zero_grad(&mut self) {
        self.gradients = Array2::zeros(self.gradients.raw_dim());
    }
}

pub trait HasGrad<T> {
    /// Differentiable function
    fn f(&self, x: T) -> T;
    /// Derivative of function
    fn d_f(&self, x: T) -> T;
}

pub trait Loss<T: Clone> {
    fn l(&self, output: &Array1<f64>, target: &T) -> f64;
    fn d_l(&self, output: &Array1<f64>, target: &T) -> Array1<f64>;
}

pub struct Optimizer<T> {
    pub loss: Box<dyn Loss<T>>,
    pub network: Network,
    pub max_gradient: Option<f64>,
}

const EMPTY_ACTIVATIONS_ERROR_MESSAGE: &str = "Fatal: Vec [activations] was empty, but it should always have at least one member (logic error)";

impl<T: Clone> Optimizer<T> {
    /// Apply the network to an input and update gradients, returning the output and loss as a
    /// tuple
    pub fn apply(&mut self, input: &Array1<f64>, target: &T) -> (Array1<f64>, f64) {
        let num_layers = self.network.layers.len();
        if num_layers == 0 {
            return (input.clone(), self.loss.l(input, target));
        }

        // Forward pass
        let mut linear_outputs: Vec<Array1<f64>> = vec![Array1::zeros(input.raw_dim())];
        let mut activations: Vec<Array1<f64>> = vec![input.clone()];
        let mut d_activations: Vec<Array1<f64>> = vec![Array1::zeros(input.raw_dim())];
        for layer in self.network.layers.iter() {
            let prev_activation = activations.last().expect(EMPTY_ACTIVATIONS_ERROR_MESSAGE);
            let linear_output = layer.weights.dot(prev_activation);
            let activation = linear_output.clone().mapv_into(|x| layer.activation.f(x));
            let d_activation = linear_output.clone().mapv_into(|x| layer.activation.d_f(x));
            linear_outputs.push(linear_output);
            activations.push(activation);
            d_activations.push(d_activation);
        }
        let output = activations.last().expect(EMPTY_ACTIVATIONS_ERROR_MESSAGE);
        let loss = self.loss.l(output, target);

        // Backward pass
        let d_l = self.loss.d_l(output, target);
        let mut loss_activation_gradients: Vec<Array1<f64>> = vec![d_l];

        for k in (1..num_layers + 1).rev() {
            let layer = &mut self.network.layers[k - 1];

            for i in 0..layer.gradients.nrows() {
                for j in 0..layer.gradients.ncols() {
                    // dbg!(loss_activation_gradients[loss_activation_gradients.len() - 1][i]);
                    // dbg!(d_activations[k][i]);
                    // dbg!(activations[activations.len() - 1][j]);
                    layer.gradients[[i, j]] += loss_activation_gradients[num_layers - k][i]
                        * d_activations[k][i]
                        * activations[k - 1][j];
                    if let Some(max_gradient) = self.max_gradient {
                        layer.gradients[[i, j]] =
                            layer.gradients[[i, j]].clamp(-max_gradient, max_gradient);
                    }
                }
            }

            let mut new_loss_activation_gradient: Array1<f64> =
                Array1::zeros(activations[k - 1].raw_dim());
            for i in 0..activations[k - 1].len() {
                for j in 0..activations[k].len() {
                    new_loss_activation_gradient[i] += loss_activation_gradients[num_layers - k][j]
                        * d_activations[k][j]
                        * self.network.layers[k - 1].weights[[j, i]];
                }
            }
            loss_activation_gradients.push(new_loss_activation_gradient);
        }

        (
            activations.pop().expect(EMPTY_ACTIVATIONS_ERROR_MESSAGE),
            loss,
        )
    }

    /// Update weights and zero the gradients
    pub fn step(&mut self, learning_rate: f64) {
        self.network.update(learning_rate);
        self.network.zero_grad();
    }
}
