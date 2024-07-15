# rust_nn

Neural network stuff in Rust. To do list for the project:

- Find a way to download datasets such as MNIST. More complicated datasets are good too.
- Separate the optimizer in a Optimizer struct (later there will be Adam, GD, SGD, etc. each of them in a separate struct)
- Update the training loop using the optimizer
- Implement dataloaders.

Side Tasks:

- Optimize matrix multiplication and vectorize other functions in `functions.rs`
- Find a way to use GPUs (I guess this is hard)

