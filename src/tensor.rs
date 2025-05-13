use anyhow::{Error, Result};
use num_traits::NumAssign;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::fmt::{self, Debug, Display, Formatter};
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A GradFn maps the output gradient (Array) and the input values (Vec<Tensor>) to the input
/// gradients (Vec<Array>).
type GradFn<A> = Box<dyn Fn(Array<A>, &Vec<Tensor<A>>) -> Vec<Array<A>> + 'static>;

pub trait DataType:
    Clone + Display + Debug + NumAssign + Neg<Output = Self> + Sum + PartialOrd + Sync + Send + 'static
{
    fn into_f64(self) -> f64;
    fn from_f64(f: f64) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, n: Self) -> Self;
    fn exp(&self) -> Self {
        Self::from_f64(f64::exp(self.clone().into_f64()))
    }
    fn sqrt(&self) -> Self {
        self.powf(Self::from_f64(0.5))
    }
}
impl DataType for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(f: f64) -> Self {
        f as f32
    }
    fn powi(&self, n: i32) -> Self {
        Self::powi(*self, n)
    }
    fn powf(&self, n: Self) -> Self {
        Self::powf(*self, n)
    }
}
impl DataType for f64 {
    fn into_f64(self) -> f64 {
        self
    }
    fn from_f64(f: f64) -> Self {
        f
    }
    fn powi(&self, n: i32) -> Self {
        Self::powi(*self, n)
    }
    fn powf(&self, n: Self) -> Self {
        Self::powf(*self, n)
    }
}

/// The Array struct is the storage underlying the Tensor struct
#[derive(Clone)]
pub struct Array<A: DataType> {
    pub dim: Box<[usize]>,
    pub data: Arc<Vec<A>>,
}

struct BackwardNode<A: DataType> {
    pub input_tensors: Vec<Tensor<A>>,
    pub grad_fn: GradFn<A>,
    pub fn_name: String,
}

impl<A: DataType> Debug for BackwardNode<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[BackwardNode: {}]", self.fn_name)
    }
}

impl<A: DataType> BackwardNode<A> {
    pub fn new(grad_fn: GradFn<A>, input_tensors: Vec<Tensor<A>>, fn_name: &str) -> Self {
        BackwardNode {
            input_tensors,
            grad_fn,
            fn_name: fn_name.to_string(),
        }
    }
}

impl<A: DataType> Array<A> {
    pub fn new(dim: Box<[usize]>, data: Arc<Vec<A>>) -> Self {
        assert!(!dim.is_empty(), "Array dimensions cannot be empty");
        assert_eq!(
            data.len(),
            dim.iter().product::<usize>(),
            "Data length does not match dimensions"
        );

        Array { dim, data }
    }

    pub fn zeros(dim: &[usize]) -> Self {
        Array::new(dim.into(), Arc::new(vec![A::zero(); dim.iter().product()]))
    }

    pub fn zeros_like(&self) -> Self {
        Array::new(self.dim.clone(), Arc::new(vec![A::zero(); self.data.len()]))
    }

    pub fn shape(&self) -> &[usize] {
        &self.dim
    }

    // Math operations
    pub fn t(&self) -> Self {
        let dim: Vec<usize> = self.dim.iter().rev().cloned().collect();
        let mut data = vec![A::zero(); self.data.len()];
        for i in 0..self.dim[0] {
            for j in 0..self.dim[1] {
                data[j * self.dim[0] + i] = self.data[i * self.dim[1] + j].clone();
            }
        }
        Array::new(dim.into(), Arc::new(data))
    }

    pub fn gelu(&self) -> Self {
        let normal = Normal::standard();
        let result_array = self
            .data
            .iter()
            .map(|x| x.clone() * A::from_f64(normal.cdf(x.clone().into_f64())))
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Self {
        let result_array = self
            .data
            .iter()
            .map(|x| x.clone().powf(A::from_f64(0.5)))
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }

    /// Element-wise exp
    pub fn exp(&self) -> Self {
        let result_array = self
            .data
            .iter()
            .map(|x| A::from_f64(f64::exp(x.clone().into_f64())))
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }

    /// Element-wise log
    pub fn log(&self) -> Self {
        let result_array = self
            .data
            .iter()
            .map(|x| A::from_f64(f64::ln(x.clone().into_f64())))
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }

    /// Index
    pub fn index(&self, index: &[usize]) -> Self {
        let offset = get_offset_from_index(index, &self.dim);
        Array::new(Box::new([1]), Arc::new(vec![self.data[offset].clone()]))
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.dim.iter().product()
    }
}

fn get_offset_from_index(index: &[usize], dim: &[usize]) -> usize {
    assert_eq!(
        index.len(),
        dim.len(),
        "Index length must match tensor dimensions"
    );

    let mut offset = 0;
    let mut prod = 1;
    for (i, &d) in index.iter().rev().enumerate() {
        assert!(d < dim[dim.len() - 1 - i], "Index out of bounds");
        offset += d * prod;
        prod *= dim[dim.len() - 1 - i];
    }
    offset
}

impl<A: DataType> Debug for Array<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let data = &self.data;
        let dims = &self.dim;
        format_nested(f, data, dims, 0, 0)
    }
}

#[derive(Clone)]
pub struct Tensor<A: DataType>(Arc<RwLock<InnerTensor<A>>>);

struct InnerTensor<A: DataType> {
    array: Array<A>,
    require_grad: bool,
    grad: Option<Array<A>>,
    backward_node: Option<BackwardNode<A>>,
}

impl<A: DataType> InnerTensor<A> {
    fn _new(
        array: Array<A>,
        require_grad: bool,
        grad: Option<Array<A>>,
        backward_node: Option<BackwardNode<A>>,
    ) -> Self {
        InnerTensor {
            array,
            require_grad,
            grad,
            backward_node,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }
}

impl<A: DataType + Display> Display for Tensor<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let inner_tensor = self.inner_tensor();
        let data = &inner_tensor.array.data;
        let dims = &inner_tensor.array.dim;

        if dims.is_empty() {
            return write!(f, "{}", data[0]);
        }

        format_nested(f, data, dims, 0, 0)?;

        // Add shape and dtype information
        write!(
            f,
            "\nshape: [{dims}], dtype: {}",
            self.data_type(),
            dims = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn format_nested<A: Debug + Display>(
    f: &mut Formatter<'_>,
    data: &[A],
    dims: &[usize],
    depth: usize,
    offset: usize,
) -> fmt::Result {
    if depth == dims.len() - 1 {
        // Last dimension - print a single row
        write!(f, "[")?;
        for i in 0..dims[depth] {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", data[offset + i])?;
        }
        write!(f, "]")
    } else {
        // Higher dimensions - recurse for each slice
        write!(f, "[")?;
        let slice_size = dims[depth + 1..].iter().product::<usize>();
        for i in 0..dims[depth] {
            if i > 0 {
                write!(f, ",\n{}", "  ".repeat(depth + 1))?;
            }
            format_nested(f, data, dims, depth + 1, offset + i * slice_size)?;
        }
        write!(f, "]")
    }
}

impl<A: DataType> Debug for Tensor<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let inner_tensor = self.inner_tensor();
        let data = &inner_tensor.array.data;
        let dims = &inner_tensor.array.dim;
        format_nested(f, data, dims, 0, 0)?;

        if let Some(grad) = &inner_tensor.grad {
            write!(f, "\ngrad:\n{:?}", grad)?;
        }

        if let Some(backward_node) = &inner_tensor.backward_node {
            write!(f, "\nbackward_node: {:?}", backward_node)?;
        }

        Ok(())
    }
}

impl<A: DataType> Tensor<A> {
    // Accessors

    /// Acquire a read lock on the inner tensor
    fn inner_tensor(&self) -> RwLockReadGuard<'_, InnerTensor<A>> {
        self.0.read().unwrap()
    }

    /// Acquire a write lock on the inner tensor
    fn inner_tensor_mut(&self) -> RwLockWriteGuard<'_, InnerTensor<A>> {
        self.0.write().unwrap()
    }

    /// Get the gradient of the tensor
    pub fn grad(&self) -> Option<Array<A>> {
        self.inner_tensor().grad.clone()
    }

    /// Return an Arc to the raw data of the tensor
    pub fn data(&self) -> Arc<Vec<A>> {
        self.inner_tensor().array.data.clone()
    }

    pub fn array(&self) -> Array<A> {
        self.inner_tensor().array.clone()
    }

    /// Get the inputs used to calculate the tensor
    pub fn inputs(&self) -> Option<Vec<Tensor<A>>> {
        self.inner_tensor().backward_node.as_ref().map(|node| {
            node.input_tensors
                .iter()
                .map(|tensor| tensor.clone())
                .collect()
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        self.inner_tensor().array.dim.to_vec()
    }

    /// Get the data type of the tensor as a String
    pub fn data_type(&self) -> String {
        std::any::type_name::<A>().to_string()
    }

    // Constructors

    /// Make a tensor from an inner tensor
    fn wrap(inner_tensor: InnerTensor<A>) -> Self {
        Tensor(Arc::new(RwLock::new(inner_tensor)))
    }

    /// Make a tensor of all zeros with a given shape
    pub fn zeros(dim: &[usize], require_grad: bool) -> Self {
        let data = Arc::new(vec![A::zero(); dim.iter().product()]);
        Tensor::wrap(InnerTensor::_new(
            Array::new(dim.into(), data),
            require_grad,
            None,
            None,
        ))
    }

    /// Make a tensor of all zeros with the same shape as another tensor
    pub fn zeros_like(&self) -> Self {
        let inner_tensor = self.inner_tensor();
        let data = Arc::new(vec![A::zero(); inner_tensor.array.data.len()]);
        Tensor::wrap(InnerTensor::_new(
            Array::new(inner_tensor.array.dim.clone(), data),
            inner_tensor.require_grad,
            None,
            None,
        ))
    }

    /// Make a tensor of all ones with a given shape
    pub fn ones(dim: &[usize], require_grad: bool) -> Self {
        let data = Arc::new(vec![A::one(); dim.iter().product()]);
        Tensor::wrap(InnerTensor::_new(
            Array::new(dim.into(), data),
            require_grad,
            None,
            None,
        ))
    }

    /// Make a tensor of all ones with the same shape as another tensor
    pub fn ones_like(&self) -> Self {
        let inner_tensor = self.inner_tensor();
        let data = Arc::new(vec![A::one(); inner_tensor.array.data.len()]);
        Tensor::wrap(InnerTensor::_new(
            Array::new(inner_tensor.array.dim.clone(), data),
            inner_tensor.require_grad,
            None,
            None,
        ))
    }

    /// Make a tensor of random values with a given shape
    pub fn random(dim: &[usize], require_grad: bool) -> Self {
        let data = Arc::new(
            vec![A::zero(); dim.iter().product()]
                .iter()
                .map(|_| A::from_f64(rand::random::<f64>()))
                .collect::<Vec<_>>(),
        );
        Tensor::wrap(InnerTensor::_new(
            Array::new(dim.into(), data),
            require_grad,
            None,
            None,
        ))
    }

    /// Make a tensor from a vector of data
    pub fn new(dim: &[usize], data: Arc<Vec<A>>, require_grad: bool) -> Self {
        Tensor::wrap(InnerTensor::_new(
            Array::new(dim.into(), data),
            require_grad,
            None,
            None,
        ))
    }

    // Grad Operations

    /// Backward pass, should only be called on scalar tensors
    pub fn backward(&mut self) -> Result<()> {
        {
            if !self.inner_tensor().require_grad {
                return Err(Error::msg(
                    "`backwards` should only be called on tensors with `require_grad` set to `true`",
                ));
            };

            let mut inner_tensor = self.inner_tensor_mut();
            let shape = inner_tensor.shape();
            if shape.iter().product::<usize>() != 1 {
                return Err(Error::msg(
                    "`backwards` should only be called on scalar tensors",
                ));
            }
            if !inner_tensor.require_grad {
                return Err(Error::msg(
                "`backwards` should only be called on tensors with `require_grad` set to `true`",
            ));
            }
            inner_tensor.grad = Some(Array::new(shape.into(), Arc::new(vec![A::one()])));
        }
        self.calculate_grad_graph()?;
        Ok(())
    }

    /// Recursively calculate the gradient graph, assuming current grad has been set
    fn calculate_grad_graph(&mut self) -> Result<()> {
        let input_tensors_with_grads = {
            let inner_tensor = self.inner_tensor();
            if !inner_tensor.require_grad {
                return Ok(());
            }
            let self_grad = match &inner_tensor.grad {
                Some(grad) => grad.clone(),
                None => {
                    return Err(Error::msg(
                        "Gradient is not set. Please set the gradient before calling `calculate_grad_graph`",
                    ));
                }
            };

            let (input_tensors, grad_fn) = if let Some(backward_node) = &inner_tensor.backward_node
            {
                // eprintln!("Backward node: {:?}", backward_node);
                let input_tensors = &backward_node.input_tensors;
                let grad_fn = &backward_node.grad_fn;
                (input_tensors, grad_fn)
            } else {
                return Ok(());
            };

            let grads = grad_fn(self_grad, input_tensors);
            let mut input_tensors_with_grads = Vec::new();
            for (input_tensor, grad) in input_tensors.iter().zip(grads) {
                let mut input_inner_tensor = input_tensor.inner_tensor_mut();
                if input_inner_tensor.require_grad {
                    // TODO: This is a bit inefficient, we should be able to avoid making a new
                    // Array here, but we need to refactor the Array struct to allow for interior
                    // mutability
                    input_inner_tensor.grad = match &input_inner_tensor.grad {
                        Some(g) => Some(g + &grad),
                        None => Some(grad),
                    };
                    input_tensors_with_grads.push(input_tensor.clone());
                }
            }
            input_tensors_with_grads
        };
        for mut input_tensor in input_tensors_with_grads {
            input_tensor.calculate_grad_graph()?;
        }

        Ok(())
    }

    /// Manually set the gradient of the tensor
    pub fn set_grad(&mut self, grad: Array<A>) {
        let mut inner_tensor = self.inner_tensor_mut();
        inner_tensor.grad = Some(grad);
    }

    /// Update the tensor with the given learning rate
    pub fn update(&mut self, lr: A, grad: &Array<A>) {
        let mut inner_tensor = self.inner_tensor_mut();
        let shape = inner_tensor.shape();
        assert_eq!(
            shape, &*grad.dim,
            "Shape mismatch between tensor and gradient"
        );
        inner_tensor.array.data = Arc::new(
            inner_tensor
                .array
                .data
                .iter()
                .zip(grad.data.iter())
                .map(|(x, g)| x.clone() - lr.clone() * g.clone())
                .collect(),
        );
    }

    /// Set the gradient of the tensor to zero
    pub fn zero_grad(&mut self) {
        let mut inner_tensor = self.inner_tensor_mut();
        inner_tensor.grad = None;
        inner_tensor.backward_node = None;
    }

    // Math operations

    /// Transpose the tensor
    pub fn t(&self) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.t();
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, _inputs| vec![self_grad.t()]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "t"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Gaussian Error Linear Unit
    pub fn gelu(&self) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.gelu();
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Gelu grad_fn requires 1 input");
                let normal = Normal::standard();
                let input_inner = inputs[0].inner_tensor();
                let deriv_data: Vec<A> = input_inner
                    .array
                    .data
                    .iter()
                    .map(|x| {
                        let x_f64 = x.clone().into_f64();
                        let cdf = normal.cdf(x_f64);
                        let pdf = normal.pdf(x_f64);
                        let deriv = cdf + x_f64 * pdf;
                        A::from_f64(deriv)
                    })
                    .zip(self_grad.data.iter())
                    .map(|(deriv, self_grad)| deriv * self_grad.clone())
                    .collect();
                let deriv_array = Array::new(input_inner.shape().into(), Arc::new(deriv_data));
                vec![deriv_array]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "gelu"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Sum all the elements in the tensor
    pub fn sum_all(&self) -> Self {
        let self_inner = self.inner_tensor();
        let sum = self_inner
            .array
            .data
            .iter()
            .fold(A::zero(), |acc, x| acc + x.clone());

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Sum all grad_fn requires 1 input");
                let shape = inputs[0].shape();
                let len = shape.iter().product();
                vec![Array::new(
                    shape.into(),
                    Arc::new(vec![self_grad.data[0].clone(); len]),
                )]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "sum_all"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            Array::new(Box::new([1]), Arc::new(vec![sum])),
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Sum across the rows of a 2-D tensor, keeping dims
    /// Yeah it's jank but I'm not implementing views right now
    /// It's dumb to repeat the sums but w/e
    pub fn sum_rows(&self) -> Self {
        let self_inner = self.inner_tensor();
        let num_rows = self_inner.shape()[0];
        let num_cols = self_inner.shape()[1];
        let row_sums: Vec<_> = self_inner
            .array
            .data
            .chunks(num_cols)
            .flat_map(|row| std::iter::repeat_n(row.iter().cloned().sum::<A>(), num_cols))
            .collect();

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Sum rows grad_fn requires 1 input");
                let num_cols = inputs[0].shape()[1];
                let self_grad_row_sums: Vec<_> = self_grad
                    .data
                    .chunks(num_cols)
                    .flat_map(|chunk| {
                        std::iter::repeat_n(chunk.iter().cloned().sum::<A>(), num_cols)
                    })
                    .collect();
                vec![Array::new(
                    inputs[0].shape().into(),
                    Arc::new(self_grad_row_sums),
                )]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "sum_rows"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            Array::new(Box::new([num_rows, num_cols]), Arc::new(row_sums)),
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Sum across the columns of a 2-D tensor, keeping dims
    pub fn sum_cols(&self) -> Self {
        let self_inner = self.inner_tensor();
        let num_rows = self_inner.shape()[0];
        let num_cols = self_inner.shape()[1];

        let mut col_sums: Vec<_> = vec![A::zero(); num_rows * num_cols];
        for (n, v) in self_inner.array.data.iter().enumerate() {
            col_sums[n % num_cols] += v.clone();
        }
        for i in 1..num_rows {
            let row_start = i * num_cols;
            for j in 0..num_cols {
                col_sums[row_start + j] = col_sums[j].clone();
            }
        }

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Sum cols grad_fn requires 1 input");
                let shape = inputs[0].shape();
                let num_rows = shape[0];
                let num_cols = shape[1];

                let mut self_grad_col_sums = vec![A::zero(); num_rows * num_cols];
                for (n, v) in self_grad.data.iter().enumerate() {
                    self_grad_col_sums[n % num_cols] += v.clone();
                }
                for i in 1..num_rows {
                    let row_start = i * num_cols;
                    for j in 0..num_cols {
                        self_grad_col_sums[row_start + j] = self_grad_col_sums[j].clone();
                    }
                }
                vec![Array::new(shape.into(), Arc::new(self_grad_col_sums))]
            });

            Some(BackwardNode::new(grad_fn, vec![self.clone()], "sum_cols"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            Array::new(Box::new([num_rows, num_cols]), Arc::new(col_sums)),
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let self_inner = self.inner_tensor();
        let other_inner = other.inner_tensor();
        let require_grad = self_inner.require_grad || other_inner.require_grad;
        let result_array = (&self_inner.array % &other_inner.array)?;

        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                vec![
                    (&self_grad % &inputs[1].array().t()).unwrap(),
                    (&inputs[0].array().t() % &self_grad).unwrap(),
                ]
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone(), other.clone()],
                "mat_mul",
            ))
        } else {
            None
        };

        Ok(Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        )))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.sqrt();
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Sqrt grad_fn requires 1 input");
                let input_inner = inputs[0].inner_tensor();
                let deriv_data: Vec<A> = input_inner
                    .array
                    .data
                    .iter()
                    .map(|x| x.clone().powf(A::from_f64(-0.5)) * A::from_f64(0.5))
                    .zip(self_grad.data.iter())
                    .map(|(deriv, self_grad)| deriv * self_grad.clone())
                    .collect();
                let deriv_array = Array::new(input_inner.shape().into(), Arc::new(deriv_data));
                vec![deriv_array]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "sqrt"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Element-wise square
    pub fn square(&self) -> Self {
        return self * self;
    }

    /// Element-wise exp
    pub fn exp(&self) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.exp();
        let result_cache = result_array.clone();
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(move |self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Exp grad_fn requires 1 input");
                let input_inner = inputs[0].inner_tensor();
                let deriv_data: Vec<A> = result_cache
                    .data
                    .iter()
                    .zip(self_grad.data.iter())
                    .map(|(x, self_grad)| x.clone() * self_grad.clone())
                    .collect();
                let deriv_array = Array::new(input_inner.shape().into(), Arc::new(deriv_data));
                vec![deriv_array]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "exp"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Element-wise log
    pub fn log(&self) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.log();
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Log grad_fn requires 1 input");
                let input_inner = inputs[0].inner_tensor();
                let deriv_data: Vec<A> = input_inner
                    .array
                    .data
                    .iter()
                    .zip(self_grad.data.iter())
                    .map(|(x, self_grad)| self_grad.clone() / x.clone())
                    .collect();
                let deriv_array = Array::new(input_inner.shape().into(), Arc::new(deriv_data));
                vec![deriv_array]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "log"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Index
    pub fn index(&self, index: &[usize]) -> Self {
        let self_inner = self.inner_tensor();
        let result_array = self_inner.array.index(index);

        let require_grad = self_inner.require_grad;
        let index_clone = index.to_vec();
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(move |self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Index grad_fn requires 1 input");
                let input = &inputs[0];
                let shape = input.shape();
                let mut out_grad = vec![A::zero(); shape.iter().product()];
                out_grad[get_offset_from_index(&index_clone, &shape)] = self_grad.data[0].clone();

                vec![Array::new(shape.into(), Arc::new(out_grad))]
            });
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "index"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Col-wise max, keeping dims
    /// Doesn't support gradients
    pub fn max_cols(&self) -> Self {
        let inner = self.inner_tensor();
        let num_rows = inner.shape()[0];
        let num_cols = inner.shape()[1];

        let mut col_maxes: Vec<A> = vec![A::from_f64(f64::NEG_INFINITY); num_rows * num_cols];
        for (n, v) in inner.array.data.iter().enumerate() {
            let col = n % num_cols;
            if *v > col_maxes[col] {
                col_maxes[col] = v.clone();
            }
        }

        for i in 1..num_rows {
            let row_start = i * num_cols;
            for j in 0..num_cols {
                col_maxes[row_start + j] = col_maxes[j].clone();
            }
        }

        Tensor::new(&[num_rows, num_cols], Arc::new(col_maxes), false)
    }

    /// Row-wise max, keeping dims
    /// Doesn't support gradients
    pub fn max_rows(&self) -> Self {
        let inner = self.inner_tensor();
        let num_rows = inner.shape()[0];
        let num_cols = inner.shape()[1];

        let mut row_maxes: Vec<A> = vec![A::from_f64(f64::NEG_INFINITY); num_rows * num_cols];
        for (n, v) in inner.array.data.iter().enumerate() {
            let row = n / num_cols;
            if *v > row_maxes[row * num_cols] {
                row_maxes[row * num_cols] = v.clone();
            }
        }

        for j in 1..num_cols {
            for i in 0..num_rows {
                row_maxes[i * num_cols + j] = row_maxes[i * num_cols].clone();
            }
        }

        Tensor::new(&[num_rows, num_cols], Arc::new(row_maxes), false)
    }

    /// Col-wise softmax
    pub fn softmax_col(&self) -> Self {
        let col_maxes = &self.max_cols();
        let stabilized_tensor = self - col_maxes;
        let exp_tensor = &stabilized_tensor.exp();
        let sum_tensor = &exp_tensor.sum_cols();
        let softmax_tensor = exp_tensor / sum_tensor;

        softmax_tensor
    }

    /// Row-wise softmax
    pub fn softmax_row(&self) -> Self {
        let row_maxes = &self.max_rows();
        let stabilized_tensor = self - row_maxes;
        let exp_tensor = stabilized_tensor.exp();
        let sum_tensor = exp_tensor.sum_rows();
        let softmax_tensor = &exp_tensor / &sum_tensor;

        softmax_tensor
    }

    /// Broadcast 1-D tensor to 2-D tensor as columns
    pub fn broadcast_col(&self, num_cols: usize) -> Self {
        let dim = self.shape();
        assert_eq!(dim.len(), 1, "Tensor must be 1-D");
        let num_rows = dim[0];
        let self_inner = self.inner_tensor();
        let result_data: Vec<_> = self_inner
            .array
            .data
            .iter()
            .flat_map(|x| std::iter::repeat_n(x, num_cols))
            .cloned()
            .collect();
        let result_array = Array::new(Box::new([num_rows, num_cols]), Arc::new(result_data));
        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 1, "Broadcast col grad_fn requires 1 input");
                let num_rows = self_grad.shape()[0];
                let num_cols = self_grad.shape()[1];
                let self_grad_row_sums: Vec<_> = self_grad
                    .data
                    .chunks(num_cols)
                    .map(|row| row.iter().cloned().sum::<A>())
                    .collect();
                let deriv_array = Array::new(Box::new([num_rows]), Arc::new(self_grad_row_sums));
                vec![deriv_array]
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone()],
                "broadcast_col",
            ))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.inner_tensor().array.numel()
    }
}

// Tensor operators with constants

impl<A: DataType> Add<A> for &Tensor<A> {
    type Output = Tensor<A>;

    /// Addition by a constant
    fn add(self, other: A) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let result_array = &self_inner.array + other;

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, _inputs| vec![self_grad.clone()]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "const add"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

impl<A: DataType> Sub<A> for &Tensor<A> {
    type Output = Tensor<A>;

    /// Subtraction by a constant
    fn sub(self, other: A) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let result_array = &self_inner.array - other;

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, _inputs| vec![self_grad.clone()]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "const sub"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

impl<A: DataType> Mul<A> for &Tensor<A> {
    type Output = Tensor<A>;

    /// Multiplication by a constant
    fn mul(self, other: A) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let result_array = &self_inner.array * other.clone();

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> =
                Box::new(move |self_grad, _inputs| vec![&self_grad.clone() * other.clone()]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "const mul"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

impl<A: DataType> Div<A> for &Tensor<A> {
    type Output = Tensor<A>;

    /// Division by a constant
    fn div(self, other: A) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let result_array = &self_inner.array / other.clone();

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> =
                Box::new(move |self_grad, _inputs| vec![&self_grad.clone() / other.clone()]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "const div"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

// Tensor operators with other tensors

impl<A: DataType> Add for &Tensor<A> {
    type Output = Tensor<A>;

    /// Matrix (element-wise) addition
    fn add(self, other: Self) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let other_inner = other.inner_tensor();
        let result_array = &self_inner.array + &other_inner.array;

        let require_grad = self_inner.require_grad || other_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 2, "Add grad_fn requires 2 inputs");
                vec![self_grad.clone(), self_grad.clone()]
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone(), other.clone()],
                "add",
            ))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

impl<A: DataType> Neg for &Tensor<A> {
    type Output = Tensor<A>;

    /// Unary negation
    fn neg(self) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let result_array = -&self_inner.array;

        let require_grad = self_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, _inputs| vec![-&self_grad]);
            Some(BackwardNode::new(grad_fn, vec![self.clone()], "neg"))
        } else {
            None
        };

        Tensor::wrap(InnerTensor {
            array: result_array,
            require_grad,
            grad: None,
            backward_node,
        })
    }
}

impl<A: DataType> Sub for &Tensor<A> {
    type Output = Tensor<A>;

    /// Matrix (element-wise) subtraction
    fn sub(self, rhs: Self) -> Self::Output {
        let self_inner = self.inner_tensor();
        let other_inner = rhs.inner_tensor();
        let result_array = &self_inner.array - &other_inner.array;

        let require_grad = self_inner.require_grad || other_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 2, "Add grad_fn requires 2 inputs");
                vec![self_grad.clone(), -&self_grad]
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone(), rhs.clone()],
                "sub",
            ))
        } else {
            None
        };

        Tensor::wrap(InnerTensor {
            array: result_array,
            require_grad,
            grad: None,
            backward_node,
        })
    }
}

impl<A: DataType> Mul for &Tensor<A> {
    type Output = Tensor<A>;

    /// Element-wise multiplication
    fn mul(self, other: Self) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let other_inner = other.inner_tensor();
        let result_array = &self_inner.array * &other_inner.array;

        let require_grad = self_inner.require_grad || other_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 2, "Mul grad_fn requires 2 inputs");
                vec![
                    &self_grad * &inputs[1].inner_tensor().array,
                    &self_grad * &inputs[0].inner_tensor().array,
                ]
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone(), other.clone()],
                "mul",
            ))
        } else {
            None
        };

        Tensor::wrap(InnerTensor {
            array: result_array,
            require_grad,
            grad: None,
            backward_node,
        })
    }
}

impl<A: DataType> Div for &Tensor<A> {
    type Output = Tensor<A>;

    /// Element-wise division
    fn div(self, other: Self) -> Tensor<A> {
        let self_inner = self.inner_tensor();
        let other_inner = other.inner_tensor();
        let result_array = &self_inner.array / &other_inner.array;

        let require_grad = self_inner.require_grad || other_inner.require_grad;
        let backward_node = if require_grad {
            let grad_fn: GradFn<A> = Box::new(|self_grad, inputs| {
                assert_eq!(inputs.len(), 2, "Div grad_fn requires 2 inputs");
                let g = &self_grad;
                let x = &inputs[0].inner_tensor().array;
                let y = &inputs[1].inner_tensor().array;
                let ret = vec![g / y, -&(&(g * x) / &(y * y))];
                ret
            });
            Some(BackwardNode::new(
                grad_fn,
                vec![self.clone(), other.clone()],
                "div",
            ))
        } else {
            None
        };

        Tensor::wrap(InnerTensor::_new(
            result_array,
            require_grad,
            None,
            backward_node,
        ))
    }
}

/// We use `Rem` (% operator) to implement matrix multiplication
impl<A: DataType> Rem for &Tensor<A> {
    type Output = Tensor<A>;

    /// Matrix multiplication
    fn rem(self, other: Self) -> Self::Output {
        self.matmul(other).unwrap()
    }
}

// Array operators with constants

impl<A: DataType> Add<A> for &Array<A> {
    type Output = Array<A>;

    fn add(self, other: A) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .map(|a| a.clone() + other.clone())
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Sub<A> for &Array<A> {
    type Output = Array<A>;

    fn sub(self, other: A) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .map(|a| a.clone() - other.clone())
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Mul<A> for &Array<A> {
    type Output = Array<A>;

    /// Multiplication by a constant
    fn mul(self, other: A) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .map(|a| a.clone() * other.clone())
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Div<A> for &Array<A> {
    type Output = Array<A>;

    /// Division by a constant
    fn div(self, other: A) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .map(|a| a.clone() / other.clone())
            .collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

// Array operators with other arrays

impl<A: DataType> Add for &Array<A> {
    type Output = Array<A>;

    fn add(self, other: Self) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Neg for &Array<A> {
    type Output = Array<A>;

    fn neg(self) -> Self::Output {
        let result_array = self.data.iter().map(|a| -a.clone()).collect();
        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Sub for &Array<A> {
    type Output = Array<A>;

    fn sub(self, other: Self) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Mul for &Array<A> {
    type Output = Array<A>;

    /// Element-wise multiplication
    fn mul(self, other: Self) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();

        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

impl<A: DataType> Div for &Array<A> {
    type Output = Array<A>;

    /// Element-wise division
    fn div(self, other: Self) -> Self::Output {
        let result_array = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() / b.clone())
            .collect();

        Array::new(self.dim.clone(), Arc::new(result_array))
    }
}

/// We use `Rem` (% operator) to implement matrix multiplication
impl<A: DataType> Rem for &Array<A> {
    type Output = Result<Array<A>>;

    /// Matrix multiplication
    fn rem(self, other: Self) -> Self::Output {
        let (result_array, output_shape) = {
            // check shapes
            let self_shape = self.shape();
            let other_shape = other.shape();
            if self_shape.len() != 2 || other_shape.len() != 2 {
                return Err(Error::msg(format!(
                    "Matrix multiplication is only supported for 2D arrays, got {:?} and {:?}",
                    self_shape, other_shape
                )));
            }
            if self_shape[1] != other_shape[0] {
                return Err(Error::msg(format!(
                    "Matrix multiplication shape error, got {:?} and {:?}, we require {:?} == {:?}",
                    self_shape, other_shape, self_shape[1], other_shape[0]
                )));
            }
            let (n, m, p) = (self_shape[0], self_shape[1], other_shape[1]);
            let output_shape = vec![n, p];

            let mut result_array = vec![A::zero(); n * p];
            for i in 0..n {
                for j in 0..p {
                    for k in 0..m {
                        result_array[p * i + j] +=
                            self.data[i * m + k].clone() * other.data[k * p + j].clone();
                    }
                }
            }

            (result_array, output_shape)
        };

        Ok(Array::new(output_shape.into(), Arc::new(result_array)))
    }
}
