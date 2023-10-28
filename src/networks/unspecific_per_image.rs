//Network #1 (unspecific)

//Unspecific in that the network structure is defined in initialisation and not hard coded
//Per image in that computations occur once for each image, then iterate over images
//(rather than a more generalised multiple images at a time via higher matrix dimensions)

//Optimised in reduced memory allocation

use ndarray::Array2;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use crate::mnist::MnistImage;
use crate::utils::{sigmoid_prime_vector, sigmoid_vector};

pub struct UnspecificPerImage {
    bias_vectors: Vec<Array2<f64>>,
    weight_matrices: Vec<Array2<f64>>,

    activation_vectors: Vec<Array2<f64>>,
    weighted_input_vectors: Vec<Array2<f64>>,

    //Every batch we train on accumulates and then averages these nabla, before ultimately mutating them in
    batch_nb: Vec<Array2<f64>>,
    batch_nw: Vec<Array2<f64>>,

    //Every image we train on stores the delta_nabla which is then later summed into batch_nabla
    image_d_nb: Vec<Array2<f64>>,
    image_d_nw: Vec<Array2<f64>>
}

impl UnspecificPerImage {
    pub fn new(structure: &[usize]) -> Box<Self> {
        let mut bias_vectors = Vec::with_capacity(structure.len());
        let mut weight_matrices = Vec::with_capacity(structure.len());

        let mut activation_vectors = Vec::with_capacity(structure.len());
        let mut weighted_input_vectors = Vec::with_capacity(structure.len());

        let mut batch_nb = Vec::with_capacity(structure.len());
        let mut batch_nw = Vec::with_capacity(structure.len());

        let mut image_d_nb = Vec::with_capacity(structure.len());
        let mut image_d_nw = Vec::with_capacity(structure.len());

        //First layer, special case
        let mut last_num_neurons = structure[0];

        bias_vectors.push(Array2::random((0,0), StandardNormal));
        weight_matrices.push(Array2::random((0,0), StandardNormal));

        batch_nb.push(Array2::zeros((0,0)));
        batch_nw.push(Array2::zeros((0,0)));

        image_d_nb.push(Array2::zeros((0,0)));
        image_d_nw.push(Array2::zeros((0,0)));

        activation_vectors.push(Array2::zeros((last_num_neurons, 1)));
        weighted_input_vectors.push(Array2::zeros((0,0)));

        for &num_neurons in &structure[1..] {

            bias_vectors.push(Array2::random((num_neurons, 1), StandardNormal));
            weight_matrices.push(Array2::random((num_neurons, last_num_neurons), StandardNormal));

            batch_nb.push(Array2::zeros((num_neurons, 1)));
            batch_nw.push(Array2::zeros((num_neurons, last_num_neurons)));

            image_d_nb.push(Array2::zeros((num_neurons, 1)));
            image_d_nw.push(Array2::zeros((num_neurons, last_num_neurons)));

            activation_vectors.push(Array2::zeros((num_neurons, 1)));
            weighted_input_vectors.push(Array2::zeros((num_neurons, 1)));

            last_num_neurons = num_neurons;
        }

        Box::new(Self {
            bias_vectors,
            weight_matrices,

            activation_vectors,
            weighted_input_vectors,

            batch_nb,
            batch_nw,

            image_d_nb,
            image_d_nw
        })
    }

    pub fn train(&mut self, training_data: &mut Vec<MnistImage>, testing_data: &[MnistImage], epochs: usize, batch_size: usize, eta: f64) {
        let mut rng = thread_rng();

        println!("Performance from random: {}%", self.evaluate(testing_data));

        for epoch in 0..epochs {
            training_data.shuffle(&mut rng);

            for batch in training_data.chunks(batch_size) {
                self.train_batch(batch, eta);
            }

            println!("Epoch {}: {}%", epoch, self.evaluate(testing_data));
        }

    }

    fn train_batch(&mut self, batch: &[MnistImage], eta: f64) {
        //Reset the batch_nabla allocations
        for a in self.batch_nb.iter_mut() { a.fill(0.0) }
        for a in self.batch_nw.iter_mut() { a.fill(0.0) }

        for image in batch {
            //Below will, within itself, mutate self.batch_nabla's after it computes image_delta_nabla's
            self.back_propagate(image);

            for (nb, dnb) in self.batch_nb.iter_mut().zip(&self.image_d_nb) {
                *nb += dnb;
            }
            for (nw, dnw) in self.batch_nw.iter_mut().zip(&self.image_d_nw) {
                *nw += dnw;
            }
        }

        let learning_rate = eta / batch.len() as f64;

        for (b, nb) in self.bias_vectors.iter_mut().zip(&self.batch_nb) {
            *b -= &nb.mapv(|v| v * learning_rate);
        }
        for (w, nw) in self.weight_matrices.iter_mut().zip(&self.batch_nw) {
            *w -= &nw.mapv(|v| v * learning_rate);
        }
    }

    fn back_propagate(&mut self, image: &MnistImage) {
        //Reset the image_delta_nabla_allocations
        for a in self.image_d_nb.iter_mut() { a.fill(0.0) }
        for a in self.image_d_nw.iter_mut() { a.fill(0.0) }

        //Feedforward
        self.feed_forward(&image.image);

        //Begin backpropagating in final layer
        {
            let layer_index = 2; //3-1
            let weighted_inputs = &self.weighted_input_vectors[layer_index];
            let activations = &self.activation_vectors[layer_index];
            let previous_activations = &self.activation_vectors[layer_index - 1];

            self.image_d_nb[layer_index] = (activations - &image.label_array) * sigmoid_prime_vector(weighted_inputs); //Final layer delta equation: in terms of inputted_weights and activations for the same layer
            self.image_d_nw[layer_index] = self.image_d_nb[layer_index].dot(&previous_activations.t()); //Nabla layer weights equation: in terms of previous layer activation and current layer delta/error. The equation on the site is never given in matrix form, but fairly logically comes down to this, including the required transposition
        }

        //Continue backpropagating
        for layer_index in (1..2).rev() {
            let next_weights = &self.weight_matrices[layer_index + 1];
            let next_delta = &self.image_d_nb[layer_index + 1];
            let current_weighted_inputs = &self.weighted_input_vectors[layer_index];
            let previous_activations = &self.activation_vectors[layer_index - 1];

            self.image_d_nb[layer_index] = next_weights.t().dot(next_delta) * sigmoid_prime_vector(current_weighted_inputs); //Delta equation in terms of 'previous' delta: in terms of next weights, next delta, current weighted inputs
            self.image_d_nw[layer_index] = self.image_d_nb[layer_index].dot(&previous_activations.t()); //Nabla layer weights equation: in terms of previous layer activation and current layer delta/error. The equation on the site is never given in matrix form, but fairly logically comes down to this, including the required transposition
        }
    }

    fn feed_forward(&mut self, input_array: &Array2<f64>) {
        for (a,&b) in self.activation_vectors[0].iter_mut().zip(input_array) {
            *a = b;
        }

        for layer_index in 1..3 {
            let b = &self.bias_vectors[layer_index];
            let w = &self.weight_matrices[layer_index];

            let input_activations = &self.activation_vectors[layer_index - 1];

            self.weighted_input_vectors[layer_index] = w.dot(input_activations) + b;
            self.activation_vectors[layer_index] = sigmoid_vector(&self.weighted_input_vectors[layer_index]);
        }

    }

    pub fn evaluate(&mut self, testing_data: &[MnistImage]) -> f64 {
        let mut correct_counter = 0;

        for image in testing_data.iter() {
            //Feedforward
            self.feed_forward(&image.image);
            let activation_vector = self.activation_vectors.last().unwrap();

            //Find what it selected
            let mut predicted_number = 0;
            let mut predicted_certainty = 0.0;
            for (index, &certainty) in activation_vector.column(0).iter().enumerate() {
                if certainty > predicted_certainty {
                    predicted_number = index as u8;
                    predicted_certainty = certainty;
                }
            }

            if predicted_number == image.label {
                correct_counter += 1;
            }
        }

        (correct_counter as f64 / testing_data.len() as f64) * 100.0
    }
}

