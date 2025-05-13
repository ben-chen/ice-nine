use crate::{Array, DataType, Tensor};

pub trait Optimizer<A: DataType> {
    fn step_num(&self) -> usize;
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD<A: DataType> {
    pub step: usize,
    lr: A,
    parameters: Vec<Tensor<A>>,
    weight_decay: A,
    momentum: A,
    rolling_average_gradient: Vec<Array<A>>,
}

impl<A: DataType> SGD<A> {
    pub fn new(lr: A, parameters: Vec<Tensor<A>>, weight_decay: A, momentum: A) -> Self {
        let rolling_average_gradient = if momentum.is_zero() {
            Vec::new()
        } else {
            parameters.iter().map(|p| p.array().zeros_like()).collect()
        };
        Self {
            step: 1,
            lr,
            parameters,
            weight_decay,
            momentum,
            rolling_average_gradient,
        }
    }
}

impl<A: DataType> Optimizer<A> for SGD<A> {
    fn step_num(&self) -> usize {
        self.step
    }

    fn step(&mut self) {
        for (p_index, p) in self.parameters.iter_mut().enumerate() {
            let grad = p.grad().unwrap();
            let decayed_grad = if !self.weight_decay.is_zero() {
                &grad + &(&p.array() * self.weight_decay.clone())
            } else {
                grad
            };

            let effective_grad = if !self.momentum.is_zero() {
                let rolling_average_gradient = self.rolling_average_gradient[p_index].clone();
                self.rolling_average_gradient[p_index] =
                    &(&rolling_average_gradient * self.momentum.clone()) + &decayed_grad;
                self.rolling_average_gradient[p_index].clone()
            } else {
                decayed_grad
            };
            p.update(self.lr.clone(), &effective_grad);
        }
        self.step += 1;
    }

    fn zero_grad(&mut self) {
        for p in self.parameters.iter_mut() {
            p.zero_grad();
        }
    }
}

pub struct AdamW<A: DataType> {
    pub step: usize,
    lr: A,
    parameters: Vec<Tensor<A>>,
    weight_decay: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    m: Vec<Array<A>>,
    v: Vec<Array<A>>,
}

impl<A: DataType> AdamW<A> {
    pub fn new(
        lr: A,
        parameters: Vec<Tensor<A>>,
        weight_decay: A,
        beta1: A,
        beta2: A,
        epsilon: A,
    ) -> Self {
        let m = parameters.iter().map(|p| p.array().zeros_like()).collect();
        let v = parameters.iter().map(|p| p.array().zeros_like()).collect();

        Self {
            step: 1,
            lr,
            parameters,
            weight_decay,
            beta1,
            beta2,
            epsilon,
            m,
            v,
        }
    }
}

impl<A: DataType> Optimizer<A> for AdamW<A> {
    fn step_num(&self) -> usize {
        self.step
    }

    /// Performs a single optimization step.
    fn step(&mut self) {
        for (p_index, p) in self.parameters.iter_mut().enumerate() {
            // decay update
            p.update(self.lr.clone() * self.weight_decay.clone(), &p.array());

            // moments update
            let grad = p.grad().unwrap_or_else(|| p.array().zeros_like());
            self.m[p_index] = &(&self.m[p_index] * self.beta1.clone())
                + &(&grad.clone() * (A::one() - self.beta1.clone()));
            self.v[p_index] = &(&self.v[p_index] * self.beta2.clone())
                + &(&((&grad.clone()) * (&grad.clone())) * (A::one() - self.beta2.clone()));

            // bias correction
            let m_hat = &self.m[p_index] / (A::one() - self.beta1.powi(self.step as i32));
            let v_hat = &self.v[p_index] / (A::one() - self.beta2.powi(self.step as i32));
            let effective_grad = &m_hat / &(&v_hat.sqrt() + self.epsilon.clone());

            // update parameter
            p.update(self.lr.clone(), &effective_grad);
        }
        self.step += 1;
    }

    fn zero_grad(&mut self) {
        for p in self.parameters.iter_mut() {
            p.zero_grad();
        }
    }
}
