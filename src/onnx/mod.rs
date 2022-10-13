use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
use anyhow::Result;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use std::marker::PhantomData;
pub mod utilities;
use std::cmp::max;
pub use utilities::*;
pub mod onnxmodel;
use log::{info, trace};
pub use onnxmodel::*;

#[derive(Clone, Debug)]
pub struct OnnxCircuit<F: FieldExt> {
    pub inputs: Vec<Tensor<i32>>,
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for OnnxCircuit<F> {
    type Config = OnnxModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let onnx_model = OnnxModel::from_arg();
        let num_advices = max(
            onnx_model.max_node_advices(),
            onnx_model.max_advices_width().unwrap(),
        );
        info!("number of advices used: {:?}", num_advices);
        let advices = VarTensor::from(Tensor::from((0..num_advices + 3).map(|_| {
            let col = meta.advice_column();
            meta.enable_equality(col);
            col
        })));

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
        let outputs = config
            .model
            .layout(config.clone(), &mut layouter, &inputs)
            .unwrap();

        trace!("laying out output in synthesize");
        let _: Vec<_> = outputs
            .iter()
            .map(|o| match o {
                ValTensor::PrevAssigned { inner: v, dims: _ } => v.enum_map(|i, x| {
                    layouter
                        .constrain_instance(x.cell(), config.public_output, i)
                        .unwrap()
                }),
                _ => panic!("should be assigned"),
            })
            .collect();
        Ok(())
    }
}
