use ndarray::{Array, Dimension};

#[inline]
pub fn sigmoid_array<D: Dimension>(z_vector: &Array<f64, D>) -> Array<f64, D> {
    //Most of the time here is in computing the negative of z_vector
    1.0 / (1.0 + (-z_vector).mapv(f64::exp))
}

#[inline]
pub fn sigmoid_prime_array<D: Dimension>(vector: &Array<f64, D>) -> Array<f64, D> {
    let a = sigmoid_array(vector);
    let b = a.mapv(|a| 1.0 - a);
    a * b
}