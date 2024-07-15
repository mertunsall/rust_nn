use crate::functions::{relu};
/* a Relu struct that has:

- no parameters
- a forward method that takes an input tensor and returns relu applied to each element
- a backward method that takes the gradient of the output and returns the gradient of the input and 
the gradients of the parameters
*/

pub struct Relu;

impl Relu {

    // initialize weight and bias normally with normal with mean 0 and std 1/sqrt(in_features)
    pub fn new() -> Self {
        Relu
    }

    // forward pass the input tensor through relu
    pub fn forward(&self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        input.iter().map(|x| relu(x)).collect()
    }

    pub fn backward(&self, grad_output: &Vec<Vec<f64>>, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut grad_input = grad_output.clone();
        for i in 0..grad_input.len() {
            for j in 0..grad_input[0].len() {
                if input[i][j] <= 0.0 {
                    grad_input[i][j] = 0.0;
                }
            }
        }
        grad_input
    }


}