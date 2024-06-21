use ndarray::{Array1, Array2, ErrorKind, ShapeError};

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn f(&self, v: Array1<f64>) -> Array1<f64> {
        self.layers.iter().fold(v, |prev_v, layer| layer.f(&prev_v))
    }

    pub fn push(&mut self, layer: Layer) -> Result<(), ShapeError> {
        if let Some(last_layer) = self.layers.last() {
            if last_layer.dims().0 != layer.dims().1 {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        self.layers.push(layer);
        Ok(())
    }

    pub fn new() -> Self {
        Network { layers: Vec::new() }
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.update(learning_rate));
    }

    pub fn zero_grad(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.zero_grad());
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

    pub fn dims(&self) -> (usize, usize) {
        self.weights.dim()
    }

    /// Update layer weights in-place
    pub fn update(&mut self, learning_rate: f64) {
        let grad_ptr: *mut Array2<_> = &mut self.gradients;
        unsafe {
            self.weights += &(learning_rate * grad_ptr.read());
        }
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
}

impl<T: Clone> Optimizer<T> {
    /// Apply the network to an input and update gradients, returning the output and loss as a
    /// tuple
    pub fn apply(&mut self, input: &Array1<f64>, target: &T) -> (Array1<f64>, f64) {
        let num_layers = self.network.layers.len();
        if num_layers == 0 {
            return (input.clone(), self.loss.l(input, target));
        }
        dbg!(input);

        // Forward pass
        let mut linear_outputs: Vec<Array1<f64>> = vec![Array1::zeros(input.raw_dim())];
        let mut activations: Vec<Array1<f64>> = vec![input.clone()];
        let mut d_activations: Vec<Array1<f64>> = vec![Array1::zeros(input.raw_dim())];
        for layer in self.network.layers.iter() {
            let prev_activation = activations.last().expect("Fatal: Vec [activations] was empty, but it should always have at least one member (logic error)");
            dbg!(&layer.weights);
            dbg!(&prev_activation);
            let linear_output = layer.weights.dot(prev_activation);
            let activation = linear_output.clone().mapv_into(|x| layer.activation.f(x));
            let d_activation = linear_output.clone().mapv_into(|x| layer.activation.d_f(x));
            linear_outputs.push(linear_output);
            activations.push(activation);
            d_activations.push(d_activation);
        }
        dbg!(&linear_outputs);
        dbg!(&activations);
        dbg!(&d_activations);
        let output = activations.last().expect("Fatal: Vec [activations] was empty, but it should always have at least one member (logic error)");
        let loss = self.loss.l(output, target);

        // Backward pass
        let d_l = self.loss.d_l(output, target);
        let mut loss_activation_gradients: Vec<Array1<f64>> = vec![d_l];

        for k in (1..num_layers + 1).rev() {
            let layer = &mut self.network.layers[k - 1];

            dbg!(&loss_activation_gradients);
            for i in 0..layer.gradients.nrows() {
                for j in 0..layer.gradients.ncols() {
                    // dbg!(loss_activation_gradients[loss_activation_gradients.len() - 1][i]);
                    // dbg!(d_activations[k][i]);
                    // dbg!(activations[activations.len() - 1][j]);
                    layer.gradients[[i, j]] = loss_activation_gradients[num_layers - k][i]
                        * d_activations[k][i]
                        * activations[k - 1][j];
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
            dbg!(&new_loss_activation_gradient);
            loss_activation_gradients.push(new_loss_activation_gradient);
        }

        (output.clone(), loss)
    }

    pub fn step(&mut self, learning_rate: f64) {
        self.network.update(learning_rate);
        self.network.zero_grad();
    }
}
