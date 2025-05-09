use crate::optimizer::Model;
use crate::tensor::{DataType, Tensor};

pub struct Sequential<A: DataType> {
    pub layers: Vec<Box<dyn Model<A>>>,
}

impl<A: DataType> Sequential<A> {
    pub fn new(layers: Vec<Box<dyn Model<A>>>) -> Self {
        Self { layers }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Model<A>>) {
        self.layers.push(layer);
    }
}

impl<A: DataType> Model<A> for Sequential<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}
