use crate::{HasGrad, Layer};
use ndarray::Array2;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Linear layer (no activation)
pub struct Linear;

impl HasGrad<f64> for Linear {
    /// Identity activation
    fn f(&self, x: f64) -> f64 {
        x
    }

    /// Derivative of identity
    fn d_f(&self, _x: f64) -> f64 {
        1.0
    }
}

impl Linear {
    /// Make new linear layer
    pub fn new_layer(weights: Array2<f64>) -> Layer {
        let gradients = Array2::zeros(weights.raw_dim());
        Layer {
            activation: Box::new(Linear {}),
            weights,
            gradients,
        }
    }
}

/// Layer with Rectified Linear Unit activation
/// Identity function for x >= 0
/// 0 for x < 0
pub struct Relu;

impl HasGrad<f64> for Relu {
    /// Rectified Linear Unit
    fn f(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            x
        }
    }

    /// Derivative of Relu
    fn d_f(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

impl Relu {
    /// Make new Relu layer
    pub fn new_layer(weights: Array2<f64>) -> Layer {
        let gradients = Array2::zeros(weights.raw_dim());
        Layer {
            activation: Box::new(Relu {}),
            weights,
            gradients,
        }
    }
}

/// Layer with Leaky Rectified Linear Unit activation
/// Same as Relu, but with a slope for x < 0
pub struct LeakyRelu {
    pub slope: f64,
}

impl HasGrad<f64> for LeakyRelu {
    /// Leaky Rectified Linear Unit
    fn f(&self, x: f64) -> f64 {
        if x < 0.0 {
            x * self.slope
        } else {
            x
        }
    }

    /// Derivative of LeakyRelu
    fn d_f(&self, x: f64) -> f64 {
        if x < 0.0 {
            self.slope
        } else {
            1.0
        }
    }
}

impl LeakyRelu {
    /// Make new LeakyRelu layer
    pub fn new_layer(weights: Array2<f64>, slope: f64) -> Layer {
        let gradients = Array2::zeros(weights.raw_dim());
        Layer {
            activation: Box::new(LeakyRelu { slope }),
            weights,
            gradients,
        }
    }
}

/// Layer with Gaussian Error Linear Unit activation
/// f(x) = x * normal_cdf(x)
pub struct Gelu {
    normal: Normal,
}

impl HasGrad<f64> for Gelu {
    /// Gaussian Error Linear Unit
    fn f(&self, x: f64) -> f64 {
        x * self.normal.cdf(x)
    }

    /// Derivative of Gelu
    fn d_f(&self, x: f64) -> f64 {
        self.normal.cdf(x) + x * self.normal.pdf(x)
    }
}

impl Gelu {
    /// Make new Gelu layer
    pub fn new_layer(weights: Array2<f64>) -> Layer {
        let gradients = Array2::zeros(weights.raw_dim());
        Layer {
            activation: Box::new(Gelu {
                normal: Normal::standard(),
            }),
            weights,
            gradients,
        }
    }
}


pub struct SelfAttention {
    query_weights: Array2<f64>,
    key_weights: Array2<f64>,
    value_weights: Array2<f64>,
}

impl SelfAttention {
}
