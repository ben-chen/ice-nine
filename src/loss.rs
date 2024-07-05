use crate::Loss;
use anyhow::Error;
use ndarray::Array1;
use rand::{distributions::Distribution, Rng};
use statrs::distribution::Normal;

pub struct LeastSquares;

impl Loss<Array1<f64>> for LeastSquares {
    fn l(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        output
            .iter()
            .zip(target)
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }

    fn d_l(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        2.0 * (output.clone() - target)
    }
}

pub struct CrossEntropy {
    pub temperature: f64,
}

/// Takes labels (class indices) as the target
impl CrossEntropy {
    const EPSILON: f64 = 1e-7;
}

impl Loss<usize> for CrossEntropy {
    /// CE Loss, [output] is logits, [target] is the label of the answer
    fn l(&self, output: &Array1<f64>, target: &usize) -> f64 {
        assert!(*target < output.len());
        let output = output / self.temperature;
        let max_logit = output.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps = output.mapv(|x| (x - max_logit).exp());
        let prob = (exps[*target] / exps.sum()).clamp(Self::EPSILON, 1.0);
        -prob.ln()
    }

    fn d_l(&self, output: &Array1<f64>, target: &usize) -> Array1<f64> {
        assert!(*target < output.len());
        let output = output / self.temperature;
        let max_logit = output.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps = output.mapv(|x| (x - max_logit).exp());
        let exp_sum = exps.sum();
        let raw_vec = (0..exps.len())
            .map(|i| {
                if i == *target {
                    (exps[*target] - exp_sum) / (self.temperature * exp_sum)
                } else {
                    exps[i] / (self.temperature * exp_sum)
                }
            })
            .collect();
        Array1::from_vec(raw_vec)
    }
}

/// Random weight scaled using He initialization
/// Normalizes so that the variance of the output doesn't explode or vanish exponentially with the
/// number of layers
/// This samples a normal distribution with mean 0 and variance 2/num_inputs_to_layer
pub fn he_random_weight<R>(num_inputs: usize, rng: &mut R) -> Result<f64, Error>
where
    R: Rng + ?Sized,
{
    let normal = Normal::new(0.0, (2.0 / num_inputs as f64).sqrt())?;
    Ok(normal.sample(rng))
}

pub fn logits_to_probs(logits: &Array1<f64>) -> Array1<f64> {
    let max_logit = logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let exp_sum = exps.sum();
    exps / exp_sum
}
