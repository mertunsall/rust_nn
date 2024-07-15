use crate::functions::{matmul, normal, transpose, add_vec_to_matrix, sum_matrix};
use rand::rngs::StdRng;
use std::fmt;
/* a Linear struct that has:

- weight, bias
- a forward method that takes an input tensor and returns the result of the matrix multiplication 
followed by the addition of the bias
- a backward method that takes the gradient of the output and returns the gradient of the input and 
the gradients of the parameters
*/

# [derive(Clone)]
pub struct Linear {
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl Linear {

    // initialize weight and bias normally with normal with mean 0 and std 1/sqrt(in_features)
    pub fn new(in_features: i64, out_features: i64, rng: &mut StdRng) -> Self {
        let std = 1.0 / (in_features as f64).sqrt();
        let weight = normal(in_features, out_features, rng, 0.0, std);
        let bias = normal(1, out_features, rng, 0.0, std)[0].clone();
        
        Linear { weight, bias }
    }

    // forward pass the input tensor through the linear layer
    // output = weight matrix * input tensor + bias
    pub fn forward(&self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = matmul(&input, &self.weight);
        output = add_vec_to_matrix(&output, &self.bias);
        output
    }

    pub fn backward(&self, grad_output: &Vec<Vec<f64>>, input: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let grad_input = matmul(&grad_output, &transpose(&self.weight));
        let grad_weight = matmul(&transpose(&input), &grad_output);
        // grad bias is sum of grad_output which is of shape (num_samples, out_features) among the samples
        let grad_bias = sum_matrix(&grad_output, 0);
        (grad_input, grad_weight, grad_bias)
    }


}

// Custom implementation of fmt::Debug for Linear
impl fmt::Debug for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Weight:")?;
        for row in &self.weight {
            for &val in row {
                write!(f, "{:.2} ", val)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Bias:")?;
        for &val in &self.bias {
            write!(f, "{:.2} ", val)?;
        }
        writeln!(f)?;
        Ok(())
    }
}