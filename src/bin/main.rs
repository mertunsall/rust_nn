
use rand::rngs::StdRng;
use rand::SeedableRng;

use rust_nn::functions::{normal, ones_like, print_matrix, print_vector, matmul, add_vec_to_matrix};
use rust_nn::linear::Linear;
//use rust_nn::relu::Relu;
use rust_nn::mseloss::MSELoss;

// initialize a tensor of batch size 2, 2 input features
// initialize a linear layer with 2 input features and 1 output feature
// forward pass the input tensor through the linear layer
// print the input and the output

// TODO: make everything seeded


fn main() {

    // set up input, batch size, input dim, output dim
    let batch_size = 1000;
    let input_dim = 50;
    let output_dim = 1; 

    // set rng
    const SEED: u64 = 41;
    let mut rng = StdRng::seed_from_u64(SEED); 

    // create the data
    let input = normal(batch_size, input_dim, &mut rng, 0.0, 1.0);
    let std = 1.0 / (input_dim as f64).sqrt();
    let weight = normal(input_dim, output_dim, &mut rng, 0.0, std);
    let bias = normal(1, output_dim, &mut rng, 0.0, std)[0].clone();
    // y = Wx + b
    let y_true = matmul(&input, &weight);
    let y_true = add_vec_to_matrix(&y_true, &bias);

    // create the layers and initialize loss
    let mut linear_layer1 = Linear::new(input_dim, output_dim, &mut rng);
    let loss_fn = MSELoss::new();

    // do forward and backward pass
    let lr = 0.002;
    for i in 0..1001 {
        let output1 = linear_layer1.forward(&input);
        let loss = loss_fn.forward(&output1, &y_true);
    
        let gradient = loss_fn.backward(&output1, &y_true);
        let grads1 = linear_layer1.backward(&gradient, &input);
        //let gradient = grads1.0;
        let gradient_weight = grads1.1;
        let gradient_bias = grads1.2;

        // update weights
        for i in 0..input_dim {
            for j in 0..output_dim {
                let idx = i as usize;
                let jdx = j as usize;
                linear_layer1.weight[idx][jdx] -= lr * gradient_weight[idx][jdx];
            }
        }

        for i in 0..output_dim {
            let idx = i as usize;
            linear_layer1.bias[idx] -= lr * gradient_bias[idx];
        }

        if i%100 == 0 {
            println!("Epoch: {}", i);
            println!("Loss: {:.2}", loss);
            println!("");
        }
    }


    /*
    println!("Input: ");
    print_matrix(&input, 2);
    println!("True Weight: ");
    print_matrix(&weight, 2);
    println!("True Bias: ");
    print_vector(&bias, 2);
    println!("True Output:");
    print_matrix(&y_true, 2);


    println!("Linear Layer:");
    println!("{:?}", linear_layer1);
    println!("Output:");
    print_matrix(&output1, 2);
    println!("Loss: {:.2}", loss);
    println!("");

    println!("Gradient:");
    print_matrix(&gradient, 2);
    println!("Gradient Weight:");
    print_matrix(&gradient_weight, 2);
    println!("Gradient Bias:");
    print_vector(&gradient_bias, 2);
    */

}
