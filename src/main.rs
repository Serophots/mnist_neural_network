mod utils;
mod mnist;
mod networks;

extern crate blas_src;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command()]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, default_value = "network2")]
        implementation: Implementation,

        #[arg(short, long, help = "Number of training cycles. One epoch cycles the entire dataset once.")]
        epochs: Option<usize>,

        #[arg(short, long, help = "How many images to train on per batch? Balance between training efficiency and computational efficiency")]
        batch_size: Option<usize>,

        #[arg(short, long, help = "Controls the rate of gradient descent. A learning rate too high may overshoot the minimum point, whilst too low may perform poorly")]
        learning_rate: Option<f64>,

        #[arg(short, long, help = "Controls the rate of regularization to prevent over fitting to the training data, resulting in poor generalisation")]
        lambda: Option<f64>,

        #[arg(short, long, help = "Specify a file_name to save network results into")]
        save_file: Option<String>
    },
    Load {
        #[arg(required = true)]
        file_name: String,
    }
}


#[derive(ValueEnum, Clone, Debug)]
enum Implementation {
    Network1,
    Network2,
}

fn main() {
    let args = Args::parse();


    match args.command.unwrap_or(Commands::Train {
        implementation: Implementation::Network2,
        epochs: None,
        batch_size: None,
        learning_rate: None,
        lambda: None,
        save_file: None,
    } ) {
        Commands::Train {
            implementation,
            epochs,
            batch_size,
            learning_rate,
            lambda,
            save_file
        } => {
            let mut training_data = mnist::load_mnist_file("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz").unwrap();
            let testing_data = mnist::load_mnist_file("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz").unwrap();

            match implementation {
                Implementation::Network1 => {
                    let mut network = networks::network1::Network1::new(&[784, 30, 10]);

                    let epochs = epochs.unwrap_or(30);
                    let batch_size = batch_size.unwrap_or(10);
                    let learning_rate = learning_rate.unwrap_or(3.0);

                    network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, learning_rate);
                },
                Implementation::Network2 => {
                    let mut network = networks::network2::Network2::new(&[784, 30, 10]);

                    let epochs = epochs.unwrap_or(30);
                    let batch_size = batch_size.unwrap_or(10);
                    let learning_rate = learning_rate.unwrap_or(0.1);
                    let lambda = lambda.unwrap_or(5.0);

                    network.train(&mut training_data, testing_data.as_slice(), epochs, batch_size, learning_rate, lambda);
                },
            };
        },
        Commands::Load {
            file_name
        } => {

        }
    }
}