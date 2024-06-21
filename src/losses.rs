use crate::grad::Loss;
use ndarray::Array1;

pub struct LeastSquaresLoss {}

impl Loss<Array1<f64>> for LeastSquaresLoss {
    fn l(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let mut sum_of_squares = 0.0;
        output
            .clone()
            .zip_mut_with(target, |a, b| sum_of_squares += (*a - *b) * (*a - *b));
        sum_of_squares
    }

    fn d_l(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        2.0 * (output.clone() - target)
    }
}

pub struct CrossEntropyLoss {
    pub temperature: f64
}

impl CrossEntropyLoss {
    const EPSILON: f64 = 1e-7;
}

impl Loss<usize> for CrossEntropyLoss {
    /// CE Loss, [output] is logits, [target] is the label of the answer
    fn l(&self, output: &Array1<f64>, target: &usize) -> f64 {
        assert!(*target < output.len());
        let output = output / self.temperature;
        let exps = output.mapv(|x| x.exp());
        let prob = (exps[*target] / exps.sum()).clamp(Self::EPSILON, 1.0);
        -prob.ln()
    }

    fn d_l(&self, output: &Array1<f64>, target: &usize) -> Array1<f64> {
        assert!(*target < output.len());
        let output = output / self.temperature;
        let exps = output.mapv(|x| x.exp());
        let exp_sum = exps.sum().clamp(Self::EPSILON, f64::INFINITY);
        let raw_vec = (0..exps.len())
            .map(|i| {
                if i == *target {
                    (exps[*target] - exp_sum)/(self.temperature*exp_sum)
                } else {
                    exps[i]/(self.temperature*exp_sum)
                }
            })
            .collect();
        Array1::from_vec(raw_vec)
    }
}

pub fn logits_to_probs(logits: &Array1<f64>) -> Array1<f64> {
    let exps = logits.mapv(|x| x.exp());
    let exp_sum = exps.sum();
    exps / exp_sum
}
