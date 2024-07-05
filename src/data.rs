use anyhow::Error;
use ndarray::Array1;

/// Iterator over inputs and targets
#[derive(Clone)]
pub struct Dataset<'a, T> {
    pub inputs: &'a [Array1<f64>],
    pub targets: &'a [T],
    len: usize,
    i: usize,
}

impl<'a, T> Dataset<'a, T> {
    pub fn new(inputs: &'a [Array1<f64>], targets: &'a [T]) -> Result<Self, Error> {
        if inputs.len() == targets.len() {
            Ok(Dataset {
                inputs,
                targets,
                len: inputs.len(),
                i: 0,
            })
        } else {
            Err(Error::msg("[inputs] and [targets] must have the same length in a Dataset"))
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T: 'a> Iterator for Dataset<'a, T> {
    type Item = (&'a Array1<f64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let item = if self.i < self.len {
            Some((&self.inputs[self.i], &self.targets[self.i]))
        } else {
            None
        };
        self.i += 1;
        item
    }
}
