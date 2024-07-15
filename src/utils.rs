use crate::functions::{matmul, normal, add_vec_to_matrix, add_matrix_to_matrix};
use rand::rngs::StdRng;

pub fn create_linear_dataset(n_points: i64, input_dim: i64, output_dim: i64, noise_std: f64, rng: &mut StdRng)
 -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) 
    {

    // create the data
    let x = normal(n_points, input_dim, rng, 0.0, 1.0);
    let std_weight = 1.0 / (input_dim as f64).sqrt();
    let weight = normal(input_dim, output_dim, rng, 0.0, std_weight);
    let bias = normal(1, output_dim, rng, 0.0, std_weight)[0].clone();
    // y = Wx + b + noise
    let mut y = matmul(&x, &weight);
    y = add_vec_to_matrix(&y, &bias);
    let noise = normal(n_points, output_dim, rng, 0.0, noise_std);
    y = add_matrix_to_matrix(&y, &noise);

    (weight, bias, x, y)
}

pub fn get_test_data(n_points: i64, weight: &Vec<Vec<f64>>, bias: &Vec<f64>, noise_std: f64, rng: &mut StdRng) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    
    let input_dim = weight.len() as i64;
    let output_dim = weight[0].len() as i64;
    let x = normal(n_points, input_dim, rng, 0.0, 1.0);
    let mut y = matmul(&x, weight);
    y = add_vec_to_matrix(&y, bias);
    let noise = normal(n_points, output_dim, rng, 0.0, noise_std);
    y = add_matrix_to_matrix(&y, &noise);
    (x, y)
}