use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};

struct MnistData {
    metadata: Vec<i32>,
    data: Vec<u8>
}

impl MnistData {
    pub fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;

        let mut cursor = Cursor::new(&contents);

        let mut metadata: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        let magic_number = cursor.read_i32::<BigEndian>()?;
        match magic_number {
            2049 => { // set labels
                let num_items = cursor.read_i32::<BigEndian>()?;
                metadata.push(num_items);
            },
            2051 => { //set images
                let num_images = cursor.read_i32::<BigEndian>()?;
                let num_rows = cursor.read_i32::<BigEndian>()?;
                let num_columns = cursor.read_i32::<BigEndian>()?;
                metadata.push(num_images);
                metadata.push(num_rows);
                metadata.push(num_columns);
            },
            _ => panic!()
        }

        cursor.read_to_end(&mut data)?;

        Ok(MnistData { metadata, data })
    }
}

#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub label_array: Array2<f64>,
    pub label: u8,
}

pub fn load_mnist_file(image_file_name: &str, label_file_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let image_data = MnistData::new(&(File::open(image_file_name)?))?;
    let label_data = MnistData::new(&(File::open(label_file_name)?))?;

    let num_images = image_data.metadata[0];
    let num_labels = label_data.metadata[0];
    let num_pixel_rows = image_data.metadata[1];
    let num_pixel_columns = image_data.metadata[2];
    let image_size = (num_pixel_rows * num_pixel_columns) as usize;

    assert_eq!(num_images, num_labels);

    let mut images: Vec<MnistImage> = Vec::with_capacity(num_images as usize);

    for i in 0..num_images as usize {
        let start_offset = i * image_size;
        let end_offset = (i+1) * image_size;
        let image_data = image_data.data[start_offset..end_offset].iter().map(|&x|  x as f64 / 255.);

        let mut label = Array2::zeros((10,1));
        label[(label_data.data[i] as usize, 0)] = 1.0;


        let a: Array1<f64> = Array1::from_iter(image_data);
        let b: Array2<f64> = a.into_shape((784,1)).unwrap();

        images.push(MnistImage {
            image: b,
            label_array: label,
            label: label_data.data[i]
        });
    }

    Ok(images)
}
