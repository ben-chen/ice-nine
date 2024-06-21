use ndarray::Array1;

/// Iterator over inputs and targets
pub struct Dataset<'a, T> {
    pub inputs: &'a [Array1<f64>],
    pub targets: &'a [T],
    i: usize,
}

impl<'a, T> Dataset<'a, T> {
    pub fn new(inputs: &'a [Array1<f64>], targets: &'a [T]) -> Self {
        Dataset {
            inputs,
            targets,
            i: 0,
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
        self.i += 1;
        item
    }
}
