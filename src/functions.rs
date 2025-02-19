use rand::prelude::*;
use rand_distr::{StandardNormal};
use rand::rngs::StdRng;

pub fn relu(x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|&x| x.max(0.0)).collect()
}

pub fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

pub fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            result[j][i] = a[i][j];
        }
    }
    result
}

// function that takes in n,m integers and a rng by reference and returns a matrix of size n x m
// with values sampled from a standard normal distribution

pub fn normal(n: i64, m: i64, rng: &mut StdRng, mu: f64, std: f64) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; m as usize]; n as usize];
    for i in 0..n {
        for j in 0..m {
            result[i as usize][j as usize] = rng.sample(StandardNormal);
            result[i as usize][j as usize] = result[i as usize][j as usize] * std + mu;
        }
    }
    result
}

pub fn add_vec_to_matrix(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + b[j];
        }
    }
    result
}

pub fn add_matrix_to_matrix(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

pub fn add_vec_to_vec(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = a[i] + b[i];
    }
    result
}

// if axis = 0 values in the same column are summed
// if axis = 1 values in the same row are summed

pub fn sum_matrix(a: &Vec<Vec<f64>>, axis: i64) -> Vec<f64> {
    let n = a.len();
    let m = a[0].len();

    let mut res: Vec<f64>;   

    if axis == 0{
        res = vec![0.0; m];
        for i in 0..n {
            for j in 0..m {
                res[j] += a[i][j];
            }
        }
    }
    else {
        res = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                res[i] += a[i][j];
            }
        }
    }

    res

}

pub fn squared_loss(y_pred: &Vec<Vec<f64>>, y_true: &Vec<Vec<f64>>) -> f64 {
    let mut loss = 0.0;
    let n = y_pred.len();
    let m = y_pred[0].len();
    for i in 0..n {
        for j in 0..m {
            loss += (y_pred[i][j] - y_true[i][j]).powi(2);
        }
    }
    loss / ((n as f64) * (m as f64))
}

pub fn logsumexp(a: &Vec<f64>) -> f64 {
    let max_val = a.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let sum: f64 = a.iter().map(|x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

pub fn crossentropyloss(logits: &Vec<Vec<f64>>, target: &Vec<Vec<f64>>) -> f64 {
    let n = logits.len();
    let m = logits[0].len();
    let mut loss = 0.0;
    for i in 0..n {
        let logsumexp_val = logsumexp(&logits[i]);
        for j in 0..m {
            loss += - target[i][j] * (logits[i][j] - logsumexp_val);
        }
    }
    loss / (n as f64)
}

pub fn mul_matrix_by_scalar(a: &Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] * scalar;
        }
    }
    result
}

pub fn mul_vec_by_scalar(a: &Vec<f64>, scalar: f64) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = a[i] * scalar;
    }
    result
}

pub fn ones_like(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    vec![vec![1.0; m]; n]
}

pub fn zeros_like(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    vec![vec![0.0; m]; n]
}

pub fn print_matrix(a: &Vec<Vec<f64>>, fp_precision: usize) {
    for row in a {
        for val in row {
            print!("{:.*} ", fp_precision, val);
        }
        println!();
    }
    println!();

}

pub fn print_vector(a: &Vec<f64>, fp_precision: usize) {
    for val in a {
        print!("{:.*} ", fp_precision, val);
    }
    println!();
}
