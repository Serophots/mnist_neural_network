mod utils;
mod mnist;
mod networks;

use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Args {
    implementation: Option<Implementation>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    lambda: Option<f64>,
}


#[derive(ValueEnum, Clone)]
enum Implementation {
    Network1,
    Network2,
}

fn main() {
    let args = Args::parse();

    let mut training_data = mnist::load_mnist_file("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz").unwrap();
    let testing_data = mnist::load_mnist_file("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz").unwrap();

    println!("Loaded mnist data");

    let epochs = args.epochs.unwrap_or(30);

    match args.implementation.unwrap_or(Implementation::Network2) {
        Implementation::Network1 => {
            let mut network = networks::network1::Network1::new(&[784, 30, 10]);

            let batch_size = args.batch_size.unwrap_or(10);
            let learning_rate = args.learning_rate.unwrap_or(3.0);

            network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, learning_rate);
        },
        Implementation::Network2 => {
            let mut network = networks::network2::Network2::new(&[784, 30, 10]);

            let batch_size = args.batch_size.unwrap_or(10);
            let learning_rate = args.learning_rate.unwrap_or(0.1);
            let lambda = args.lambda.unwrap_or(5.0);

            network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, learning_rate, lambda);
        }
    };

}