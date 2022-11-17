use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor};
use anyhow::Result;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use std::marker::PhantomData;
pub mod utilities;
pub use utilities::*;
pub mod model;
pub mod node;
use log::{info, trace};
pub use model::*;
pub use node::*;

#[derive(Clone, Debug)]
pub struct ModelCircuit<F: FieldExt> {
    pub inputs: Vec<Tensor<i32>>,
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let onnx_model = Model::from_arg();
        let num_advices = onnx_model.max_node_advices();
        info!("number of advices used: {:?}", num_advices);
        let advices = Tensor::from((0..num_advices).map(|_| {
            let col = meta.advice_column();
            meta.enable_equality(col);
            col
        }));

        onnx_model.configure(meta, advices).unwrap()
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        trace!("Setting input in synthesize");
        let inputs = self
            .inputs
            .iter()
            .map(|i| ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(i.clone())))
            .collect::<Vec<ValTensor<F>>>();
        trace!("Setting output in synthesize");
        config
            .model
            .layout(config.clone(), &mut layouter, &inputs)
            .unwrap();

        Ok(())
    }
}
