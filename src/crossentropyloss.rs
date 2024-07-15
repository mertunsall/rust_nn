use crate::functions::{crossentropyloss, logsumexp};

pub struct CrossEntropyLoss;

/*
Cross entropy loss that takes in the logits and the target and returns the loss.
Note that the logits are the output of the model BEFORE the softmax function
and the target is a one-hot encoded vector. So the forward function shoud compute
the softmax and negative log likelihood loss at the same time as this is more efficient.

*/


impl CrossEntropyLoss {

    pub fn new() -> Self {
        CrossEntropyLoss
    }

    pub fn forward(&self, input: &Vec<Vec<f64>>, target: &Vec<Vec<f64>>) -> f64 {   
        crossentropyloss(input, target)
    }

    pub fn backward(&self, input: &Vec<Vec<f64>>, target: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = input.len();
        let m = input[0].len();
        let mut grad_input = vec![vec![0.0; m]; n];
        for i in 0..n {
            let logsumexp_val = logsumexp(&input[i]);
            for j in 0..m {
                grad_input[i][j] = (input[i][j] - logsumexp_val).exp() - target[i][j];
            }
        }
        grad_input
    }


}   