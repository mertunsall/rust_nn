
use rand::rngs::StdRng;
use rand::SeedableRng;

use rust_nn::functions::{mul_matrix_by_scalar, mul_vec_by_scalar, add_matrix_to_matrix, add_vec_to_vec};
use rust_nn::linear::Linear;
use rust_nn::mseloss::MSELoss;
use rust_nn::utils::{create_linear_dataset, get_test_data};

// Example of a simple linear regression model using the rust_nn crate

fn main() {

    // set up input, batch size, input dim, output dim
    let batch_size = 500;
    let input_dim = 20;
    let output_dim = 1; 

    // set rng
    const SEED: u64 = 41;
    let mut rng = StdRng::seed_from_u64(SEED); 

    let noise_std = 0.2;
    let (weight, bias, x, y) = create_linear_dataset(batch_size, input_dim, output_dim, noise_std, &mut rng);
    let n_test = 50;
    let (x_test, y_test) = get_test_data(n_test, &weight, &bias, noise_std, &mut rng);

    // create the layers and initialize loss
    let mut linear_layer1 = Linear::new(input_dim, output_dim, &mut rng);
    let loss_fn = MSELoss::new();

    let lr = 0.002;
    for i in 0..1001 {
        // forward pass
        let output1 = linear_layer1.forward(&x);
        let loss = loss_fn.forward(&output1, &y);
    
        // backward pass
        let gradient = loss_fn.backward(&output1, &y);
        let grads = linear_layer1.backward(&gradient, &x);
        let gradient_weight = grads.1;
        let gradient_bias = grads.2;

        // update weights
        let weight_update = mul_matrix_by_scalar(&gradient_weight, -lr);
        let bias_update = mul_vec_by_scalar(&gradient_bias, -lr);
        linear_layer1.weight = add_matrix_to_matrix(&linear_layer1.weight, &weight_update);
        linear_layer1.bias = add_vec_to_vec(&linear_layer1.bias, &bias_update);

        // print loss
        if i%200 == 0 {
            println!("Epoch: {}", i);
            println!("Loss: {:.2}", loss);

            let output_test = linear_layer1.forward(&x_test);
            let loss_test = loss_fn.forward(&output_test, &y_test);
            println!("Test Loss: {:.2}", loss_test);
            println!("");
        }
    }

}
