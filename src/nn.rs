use std::fmt::Debug;

use crate::{
    tensor::{DataType, Tensor},
    Tokenizer,
};

pub trait Model<A: DataType>: Debug {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A>;

    fn parameters(&self) -> Vec<Tensor<A>>;
}

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
        let mut params = vec![];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl<A: DataType> Debug for Sequential<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sequential {{")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  layer[{}]: {:?}", i, layer)?;
        }
        write!(f, "}}")
    }
}

pub struct Gelu {}

impl Gelu {
    pub fn new() -> Self {
        Self {}
    }
}

impl<A: DataType> Model<A> for Gelu {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        x.gelu()
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        vec![]
    }
}

impl Debug for Gelu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gelu").finish()
    }
}

pub struct Linear<A: DataType> {
    pub weight: Tensor<A>,
    pub bias: Option<Tensor<A>>,
}

impl<A: DataType> Linear<A> {
    pub fn new(weight: Tensor<A>, bias: Tensor<A>) -> Self {
        Self {
            weight,
            bias: Some(bias),
        }
    }

    pub fn new_without_bias(weight: Tensor<A>) -> Self {
        Self { weight, bias: None }
    }

    pub fn zeros(input_dim: usize, output_dim: usize) -> Self {
        let weight = Tensor::zeros(&[output_dim, input_dim], true);
        Self { weight, bias: None }
    }

    pub fn random(input_dim: usize, output_dim: usize) -> Self {
        let weight = Tensor::random(&[output_dim, input_dim], true);
        Self { weight, bias: None }
    }
}

impl<A: DataType> Model<A> for Linear<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let output = &self.weight % x;
        let output = if let Some(bias) = &self.bias {
            &output + bias
        } else {
            output
        };
        output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        if let Some(bias) = &self.bias {
            vec![self.weight.clone(), bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }
}

impl<A: DataType> Debug for Linear<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("\nweight", &self.weight)
            .field("\nbias", &self.bias)
            .finish()
    }
}

pub fn cross_entropy<A: DataType>(pred_logits: &Tensor<A>, true_logits: &Tensor<A>) -> Tensor<A> {
    assert_eq!(pred_logits.shape(), true_logits.shape());
    let pred_logits = pred_logits.softmax_col();
    let true_logits = true_logits.softmax_col();
    let loss = -&(&true_logits * &pred_logits.log()).sum_all();
    loss
}

pub fn cross_entropy_labels<A: DataType>(
    pred_logits: &Tensor<A>,
    true_labels: &Vec<usize>,
) -> Tensor<A> {
    assert_eq!(pred_logits.shape()[1], true_labels.len());
    let probs = pred_logits.softmax_col();
    let mut loss = Tensor::zeros(&[1], true);
    for (i, &true_label) in true_labels.iter().enumerate() {
        let prob = probs.index(&[true_label, i]);
        loss = &loss - &(prob.log());
    }
    &loss / A::from_f64(true_labels.len() as f64)
}

pub fn cross_entropy_llm_labels<A: DataType>(
    pred_logits: &Tensor<A>,
    true_labels: &Vec<usize>,
) -> Tensor<A> {
    assert_eq!(pred_logits.shape()[1], true_labels.len());
    eprintln!("pred_logits: {:?}", pred_logits.shape());
    let vocab_file_path =
        std::path::Path::new("/Users/benchen/workspace/ice-nine/data/wikitext103_tok.json");
    let tokenizer = Tokenizer::load_from_file(vocab_file_path).expect("Failed to load tokenizer");
    eprintln!("pred_logits: {:?}", pred_logits.shape());
    let pred_probs = pred_logits.softmax_col();
    let mut loss = Tensor::zeros(&[1], true);
    for i in 0..true_labels.len() - 1 {
        let true_label = true_labels[i + 1];
        let decoded_true_label = tokenizer.decode(&[vec![true_label as u32]]);
        eprintln!("true_label, i: {:?}, {}", decoded_true_label, i);
        let pred_prob = pred_probs.index(&[true_label, i]);
        eprintln!("pred_prob: {:?}", pred_prob);
        loss = &loss - &(pred_prob.log());
    }
    &loss / A::from_f64((true_labels.len() - 1) as f64)
}
