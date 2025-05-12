use std::{fmt::Debug, sync::Arc};

use crate::{DataType, Gelu, Linear, Model, Tensor};

pub struct LayerNorm<A: DataType> {
    pub dim: usize,
    pub eps: A,
    pub weight: Tensor<A>,
    pub bias: Tensor<A>,
}

impl<A: DataType> LayerNorm<A> {
    pub fn new(dim: usize, eps: A) -> Self {
        let weight = Tensor::ones(&[dim], true);
        let bias = Tensor::zeros(&[dim], true);
        Self {
            dim,
            eps,
            weight,
            bias,
        }
    }
}

impl<A: DataType> Model<A> for LayerNorm<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let col_sums = x.sum_cols();
        let num_rows = A::from_f64(x.shape()[0] as f64);
        let num_cols = x.shape()[1];
        let col_means = &(&col_sums / num_rows.clone());
        let x_centered = &(x - col_means);
        let col_vars = &(&x_centered.square().sum_cols() / num_rows.clone()) + self.eps.clone();
        let normalized = &(x_centered / &col_vars.sqrt());
        let weight = &self.weight.broadcast_col(num_cols);
        let bias = &self.bias.broadcast_col(num_cols);
        let renormalized = &(normalized * weight) + bias;
        renormalized
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl<A: DataType> Debug for LayerNorm<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm")
            .field("\ndim", &self.dim)
            .field("\neps", &self.eps)
            .field("\nweight", &self.weight)
            .field("\nbias", &self.bias)
            .finish()
    }
}

pub struct Attention<A: DataType> {
    pub w_q: Tensor<A>,
    pub w_k: Tensor<A>,
    pub w_v: Tensor<A>,
}

impl<A: DataType> Attention<A> {
    pub fn new(w_q: Tensor<A>, w_k: Tensor<A>, w_v: Tensor<A>) -> Self {
        Self { w_q, w_k, w_v }
    }

    pub fn random(d_model: usize, d_head: usize) -> Self {
        let w_q = Tensor::random(&[d_head, d_model], true);
        let w_k = Tensor::random(&[d_head, d_model], true);
        let w_v = Tensor::random(&[d_head, d_model], true);
        Self { w_q, w_k, w_v }
    }
}

fn causal_mask<A: DataType>(size: usize) -> Tensor<A> {
    let mut mask = vec![A::from_f64(0.0); size * size];
    for i in 0..size {
        for j in 0..size {
            if i < j {
                mask[i * size + j] = A::from_f64(-1e6);
            }
        }
    }
    Tensor::new(&[size, size], Arc::from(mask), false)
}

impl<A: DataType> Model<A> for Attention<A> {
    /// We assume d_k == d_v == d_head
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let q = &self.w_q % x;
        let k = &self.w_k % x;
        let v = &self.w_v % x;
        let d_head = self.w_q.shape()[0];
        let scores = &q.t() % &k;
        let scaled_scores = &scores / (A::from_f64((d_head as f64).sqrt()));
        let seq_len = x.shape()[1];
        let causal_scores = &scaled_scores + &causal_mask(seq_len);
        let softmax_scores = causal_scores.softmax_row();
        let attention_output = &v % &softmax_scores;
        attention_output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        vec![self.w_q.clone(), self.w_k.clone(), self.w_v.clone()]
    }
}

impl<A: DataType> Debug for Attention<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Attention")
            .field("\nw_q", &self.w_q)
            .field("\nw_k", &self.w_k)
            .field("\nw_v", &self.w_v)
            .finish()
    }
}

/// Applies a LayerNorm followed by an Attention layer with residual connection
pub struct AttentionBlock<A: DataType> {
    pub layer_norm: LayerNorm<A>,
    pub attention: Attention<A>,
}

impl<A: DataType> AttentionBlock<A> {
    pub fn new(layer_norm: LayerNorm<A>, attention: Attention<A>) -> Self {
        Self {
            layer_norm,
            attention,
        }
    }

    pub fn random(d_model: usize, d_head: usize) -> Self {
        let layer_norm = LayerNorm::new(d_model, A::from_f64(1e-8));
        let attention = Attention::random(d_model, d_head);
        Self {
            layer_norm,
            attention,
        }
    }
}

impl<A: DataType> Model<A> for AttentionBlock<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let layer_norm_output = &self.layer_norm.forward(x);
        let attention_output = &self.attention.forward(layer_norm_output);
        let residual_output = x + attention_output;
        residual_output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        let mut params = vec![];
        params.extend(self.layer_norm.parameters());
        params.extend(self.attention.parameters());
        params
    }
}

impl<A: DataType> Debug for AttentionBlock<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionBlock")
            .field("\nlayer_norm", &self.layer_norm)
            .field("\nattention", &self.attention)
            .finish()
    }
}

/// Applies a LayerNorm followed by a feedforward layer with residual connection
pub struct MlpBlock<A: DataType> {
    pub layer_norm: LayerNorm<A>,
    pub linear1: Linear<A>,
    pub activation: Box<dyn Model<A>>,
    pub linear2: Linear<A>,
}

impl<A: DataType> MlpBlock<A> {
    pub fn new(
        layer_norm: LayerNorm<A>,
        linear1: Linear<A>,
        activation: Box<dyn Model<A>>,
        linear2: Linear<A>,
    ) -> Self {
        Self {
            layer_norm,
            linear1,
            activation,
            linear2,
        }
    }

    pub fn random(d_model: usize, d_ff: usize) -> Self {
        let layer_norm = LayerNorm::new(d_model, A::from_f64(1e-8));
        let linear1 = Linear::random(d_model, d_ff);
        let activation = Box::new(Gelu::new());
        let linear2 = Linear::random(d_ff, d_model);
        Self {
            layer_norm,
            linear1,
            activation,
            linear2,
        }
    }
}

impl<A: DataType> Model<A> for MlpBlock<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let layer_norm_output = &self.layer_norm.forward(x);
        let linear1_output = &self.linear1.forward(layer_norm_output);
        let activation_output = &self.activation.forward(&linear1_output);
        let linear2_output = &self.linear2.forward(&activation_output);
        let residual_output = x + linear2_output;
        residual_output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        let mut params = vec![];
        params.extend(self.layer_norm.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.activation.parameters());
        params.extend(self.linear2.parameters());

        params
    }
}

impl<A: DataType> Debug for MlpBlock<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlpBlock")
            .field("\nlayer_norm", &self.layer_norm)
            .field("\nlinear1", &self.linear1)
            .field("\nactivation", &self.activation)
            .field("\nlinear2", &self.linear2)
            .finish()
    }
}

pub struct TransformerBlock<A: DataType> {
    pub attention_block: AttentionBlock<A>,
    pub mlp_block: MlpBlock<A>,
}

impl<A: DataType> TransformerBlock<A> {
    pub fn new(attention_block: AttentionBlock<A>, mlp_block: MlpBlock<A>) -> Self {
        Self {
            attention_block,
            mlp_block,
        }
    }

    pub fn random(d_model: usize, d_head: usize, d_ff: usize) -> Self {
        let attention_block = AttentionBlock::random(d_model, d_head);
        let mlp_block = MlpBlock::random(d_model, d_ff);
        Self {
            attention_block,
            mlp_block,
        }
    }
}

impl<A: DataType> Model<A> for TransformerBlock<A> {
    fn forward(&self, x: &Tensor<A>) -> Tensor<A> {
        let attention_output = &self.attention_block.forward(x);
        let mlp_output = self.mlp_block.forward(attention_output);
        mlp_output
    }

    fn parameters(&self) -> Vec<Tensor<A>> {
        let mut params = vec![];
        params.extend(self.attention_block.parameters());
        params.extend(self.mlp_block.parameters());
        params
    }
}

impl<A: DataType> Debug for TransformerBlock<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerBlock")
            .field("\nattention_block", &self.attention_block)
            .field("\nmlp_block", &self.mlp_block)
            .finish()
    }
}
