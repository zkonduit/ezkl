use tract_onnx::prelude::*;// use tract_hir::internal::Model;

fn main() {
    // Path to your ONNX model file
    let model_path = "examples/onnx/mnist_gan_copy/network.onnx";
    // &mut std::fs::File::open(model)?
    // Load the model
    let model = tract_onnx::onnx().model_for_path(model_path).unwrap();
    
    // save_model_to_onnx(&model, "optimized_model.onnx")?;

    println!("\n Model: {:?}", model);

    // let saved_model_path = "tract_model.onnx";
    // let model = model.into_typed().unwrap();
    // let mut file = File::create(saved_model_path)?;
    // let proto = model.to_onnx()?;
    // file.write_all(&proto.write_to_bytes()?)?;


    let runnable_model = model.into_runnable().unwrap();
    
    println!("\n Runnable Model: {:?}", runnable_model);
}
