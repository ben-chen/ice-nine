use anyhow::Error;
use ndarray::Array1;

/// Iterator over inputs and targets
pub struct Dataset<'a, T> {
    pub inputs: &'a [Array1<f64>],
    pub targets: &'a [T],
    i: usize,
    len: usize,
}

impl<'a, T> Dataset<'a, T> {
    pub fn new(inputs: &'a [Array1<f64>], targets: &'a [T]) -> Result<Self, Error> {
        if inputs.len() == targets.len() {
            Ok(Dataset {
                inputs,
                targets,
                i: 0,
                len: inputs.len(),
            })
        } else {
            Err(Error::msg("[inputs] and [targets] must have the same length in a Dataset"))
        }
    }
}

impl<'a, T: 'a> Iterator for Dataset<'a, T> {
    type Item = (&'a Array1<f64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let item = if self.i < self.inputs.len() && self.i < self.targets.len() {
            Some((&self.inputs[self.i], &self.targets[self.i]))
        } else {
            None
        };
        self.i = (self.i + 1) % self.len;
        item
    }
}
