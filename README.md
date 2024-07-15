# rust_nn

Neural network stuff in Rust. To do list for the project:

- Code Matrix and Vector types so you don't have to loop every time you want to add these (potentially templated to use comfortably with f64, f32, etc etc)
- Overload operations so that you can easily do multiplication and addition across Matrix, Vec, and Float
- Overload printing of Vector and Matrix with the functions from `functions.rs`
- Restructure the whole codebase with these types before it's too late
- Code an update method for Linear instead of directly accessing its weights
- Separate the data generation process in separate file
- Find a way to download datasets such as MNIST
- Separate the optimizer in a Optimizer struct (later there will be Adam, GD, SGD, etc etc.)
- Make the training loop nicer using the optimizer. 
