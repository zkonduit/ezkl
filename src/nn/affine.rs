use crate::nn::io::*;
use crate::nn::kernel::*;
use crate::tensor::{Tensor, TensorType};
use crate::tensor_ops::*;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt, const IN: usize, const OUT: usize> {
    // kernel is weights and biases concatenated
    pub kernel: KernelConfig<F>,
    pub input: IOConfig<F>,
    pub output: IOConfig<F>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const IN: usize, const OUT: usize> Affine1dConfig<F, IN, OUT> {
    // composable_configure takes the input tensor as an argument, and completes the advice by generating new for the rest
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        kernel: Tensor<ParamType>,
        advices: Tensor<Column<Advice>>,
    ) -> Self {

        let mut config = Self {
            selector: meta.selector(),
            kernel: KernelConfig::configure(meta, kernel, &[OUT, IN]),
            // add 1 to incorporate bias !
            input: IOConfig::configure(meta, advices.get_slice(&[0..1]), &[1, IN]),
            output: IOConfig::configure(meta, advices.get_slice(&[1..2]), &[1, OUT]),
            _marker: PhantomData,
        };

        meta.create_gate("affine", |meta| {
            let selector = meta.query_selector(config.selector);
            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = config.output.query(meta, 0);
            // Now we compute the linear expression,  and add it to constraints
            let witnessed_output = expected_output.enum_map(|i, _| {
                let mut c = Expression::Constant(<F as TensorType>::zero().unwrap());
                for j in 0..IN {
                    c = c + config.kernel.params[i].query(meta, Rotation(j as i32))
                        * meta.query_advice(advices[0], Rotation(j as i32));
                }
                c
                // add the bias
            });

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        kernel: Tensor<Value<F>>,
        input: Tensor<Value<F>>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "assign image and kernel",
            |mut region| {
                let offset = 0;
                self.selector.enable(&mut region, offset)?;

                let output = matmul(kernel.clone(), input.clone());
                self.input.assign(&mut region, offset, input.clone());
                self.kernel.assign(&mut region, offset, kernel.clone());
                Ok(self.output.assign(&mut region, offset, output))
            },
        )
    }
}
