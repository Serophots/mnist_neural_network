use ndarray::Array2;

#[inline]
pub fn sigmoid_vector(z_vector: &Array2<f64>) -> Array2<f64> {
    //Most of the time here is in computing the negative of z_vector
    1.0 / (1.0 + (-z_vector).mapv(f64::exp))
}
#[inline]
pub fn sigmoid_prime_vector(vector: &Array2<f64>) -> Array2<f64> {
    let a = sigmoid_vector(vector);
    let b = a.mapv(|a| 1.0 - a);
    a * b
}