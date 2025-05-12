#[macro_export]
macro_rules! tensor {
    // Pattern: data array and dimensions
    ([$($x:expr),* $(,)?], [$($d:expr),* $(,)?]) => {{
        let data = std::sync::Arc::new(vec![$($x),*]);
        let dims = vec![$($d),*];

        // Validate that product of dimensions matches data length
        let expected_len: usize = dims.iter().product();
        if data.len() != expected_len {
            panic!("Data length {} does not match product of dimensions {}", data.len(), expected_len);
        }

        crate::tensor::Tensor::new(&dims, data, true)
    }};

    // Pattern: data array, dimensions, and no_grad
    ([$($x:expr),* $(,)?], [$($d:expr),* $(,)?]; no_grad) => {{
        let data = std::sync::Arc::new(vec![$($x),*]);
        let dims = vec![$($d),*];

        // Validate that product of dimensions matches data length
        let expected_len: usize = dims.iter().product();
        if data.len() != expected_len {
            panic!("Data length {} does not match product of dimensions {}", data.len(), expected_len);
        }

        crate::tensor::Tensor::new(&dims, data, false)
    }};
}

// Tests
#[cfg(test)]
mod tests {
    #[test]
    fn test_scalar_tensor() {
        let t = tensor!([1.0f32], [1]);
        assert_eq!(t.shape(), vec![1]);
        assert_eq!(t.data().as_ref(), &[1.0f32]);
    }

    #[test]
    fn test_vector_tensor() {
        let t = tensor!([1.0f32, 2.0f32, 3.0f32], [3]);
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.data().as_ref(), &[1.0f32, 2.0f32, 3.0f32]);
    }

    #[test]
    fn test_matrix_tensor() {
        let t = tensor!([1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]);
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.data().as_ref(), &[1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    }

    #[test]
    fn test_tensor_with_grad() {
        let t = tensor!([1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]; no_grad);
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.data().as_ref(), &[1.0f32, 2.0f32, 3.0f32, 4.0f32]);
    }

    #[test]
    #[should_panic(expected = "Data length 3 does not match product of dimensions 4")]
    fn test_invalid_dimensions() {
        let _t = tensor!([1.0f32, 2.0f32, 3.0f32], [2, 2]);
    }
}
