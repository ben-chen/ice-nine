use crate::Loss;
use ndarray::Array1;

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

pub fn logits_to_probs(logits: &Array1<f64>) -> Array1<f64> {
    let max_logit = logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let exp_sum = exps.sum();
    exps / exp_sum
}
