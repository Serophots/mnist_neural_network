mod utils;
mod mnist;
mod networks;

use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Args {
    implementation: Option<Implementation>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    eta: Option<f64>,
}


#[derive(ValueEnum, Clone)]
enum Implementation {
    SpecificPerImage,
    UnspecificPerImage,
}

fn main() {
    let args = Args::parse();

    let mut training_data = mnist::load_mnist_file("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz").unwrap();
    let testing_data = mnist::load_mnist_file("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz").unwrap();

    println!("Loaded mnist data");


    match args.implementation.unwrap_or(Implementation::UnspecificPerImage) {
        Implementation::SpecificPerImage => {
            let mut network = networks::specific_per_image::SpecificPerImage::new();

            let epochs = args.epochs.unwrap_or(30);
            let batch_size = args.batch_size.unwrap_or(10);
            let eta = args.eta.unwrap_or(3.0);

            network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, eta)
        },
        Implementation::UnspecificPerImage => {
            let mut network = networks::unspecific_per_image::UnspecificPerImage::new(&[784, 30, 10]);

            let epochs = args.epochs.unwrap_or(30);
            let batch_size = args.batch_size.unwrap_or(10);
            let eta = args.eta.unwrap_or(3.0);

            network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, eta);
        }
    };

}