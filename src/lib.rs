pub mod model;

use crate::model::Model;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

use burn::tensor::Tensor;
use console_error_panic_hook;
use wasm_bindgen::prelude::*;
//use crate::train::run_train;

#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type Backend = burn::backend::ndarray::NdArray<f32>;

static MODEL_BYTES: &[u8] = include_bytes!("../model.bin");
#[wasm_bindgen]
pub fn run_model(image: &[f32]) -> Vec<f32> {
    console_error_panic_hook::set_once();

    #[cfg(feature = "wgpu")]
    init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default());
    let model: Model<Backend> = Model::new(&Default::default());

    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(MODEL_BYTES.to_vec(), &Default::default());

    // Load that record with the model
    let model = match record {
        Ok(rec) => model.load_record(rec),
        Err(err) => {
            log::debug!("Something went fucking wrong: {}", err);
            panic!("Oopsies")
        }
    };
    let image = Tensor::<Backend, 1>::from_floats(image, &Default::default()).reshape([1, 28, 28]);
    let image = (image - 0.1307) / 0.3081;

    let output = model.forward(image);
    let output = burn::tensor::activation::softmax(output, 1);
    let output = output.into_data();

    let mut results: Vec<f32> = Vec::new();
    for value in output.iter::<f32>() {
        results.push(value.into());
    }
    return results;
}
