use crate::grad::Loss;
use ndarray::Array1;

pub struct LeastSquaresLoss {}

impl Loss<Array1<f64>> for LeastSquaresLoss {
    fn l(&self, output: &ndarray::Array1<f64>, target: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        output
            .clone()
            .zip_mut_with(target, |a, b| sum += (*a - *b) * (*a - *b));
        sum
    }
    fn d_l(&self, output: &ndarray::Array1<f64>, target: &Array1<f64>) -> ndarray::Array1<f64> {
        2.0 * (output.clone() - target)
    }
}
