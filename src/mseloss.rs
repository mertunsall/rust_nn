use crate::functions::{squared_loss};

pub struct MSELoss;

impl MSELoss {

    pub fn new() -> Self {
        MSELoss
    }

    pub fn forward(&self, y_pred: &Vec<Vec<f64>>, y_true: &Vec<Vec<f64>>) -> f64 {
        squared_loss(y_pred, y_true)
    }


    pub fn backward(&self, y_pred: &Vec<Vec<f64>>, y_true: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = y_pred.len();
        let m = y_pred[0].len();
        let mut grad_input = vec![vec![0.0; m]; n];
        for i in 0..y_pred.len() {
            for j in 0..y_pred[0].len() {
                grad_input[i][j] = 2.0 * (y_pred[i][j] - y_true[i][j]) / ((n as f64) * (m as f64));
            }
        }
        grad_input 

    }


}