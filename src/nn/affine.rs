use super::*;
use crate::nn::io::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Assigned, ConstraintSystem, Constraints, Expression, Selector},
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt + TensorType, const IN: usize, const OUT: usize> {
    // kernel is weights and biases concatenated
    pub kernel: IOConfig<F>,
    pub input: IOConfig<F>,
    pub output: IOConfig<F>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const IN: usize, const OUT: usize> LayerConfig<F>
    for Affine1dConfig<F, IN, OUT>
{
    // composable_configure takes the input tensor as an argument, and completes the advice by generating new for the rest
    fn configure(
        meta: &mut ConstraintSystem<F>,
        kernel: ParamType,
        input: ParamType,
        output: ParamType,
    ) -> Self {
        let config = Self {
            selector: meta.selector(),
            kernel: IOConfig::configure(meta, kernel, &[OUT, IN]),
            // add 1 to incorporate bias !
            input: IOConfig::configure(meta, input, &[1, IN]),
            output: IOConfig::configure(meta, output, &[1, OUT]),
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
                    c = c + config.kernel.query_idx(meta, i, j) * config.input.query_idx(meta, 0, j)
                }
                c
                // add the bias
            });

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
        kernel: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let offset = 0;
                    self.selector.enable(&mut region, offset)?;

                    let input = self.input.assign(&mut region, offset, input.clone());
                    let weights = self.kernel.assign(&mut region, offset, kernel.clone());

                    // calculate value of output
                    let mut output: Tensor<Value<Assigned<F>>> = Tensor::new(None, &[OUT]).unwrap();
                    output = output.enum_map(|i, mut o| {
                        for (j, x) in input.iter().enumerate() {
                            o = o + x.value_field() * weights.get(&[i, j]).value_field();
                        }
                        o
                    });

                    Ok(self
                        .output
                        .assign(&mut region, offset, IOType::AssignedValue(output)))
                },
            )
            .unwrap()
    }
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
        kernels: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        self.assign(layouter, input, kernels)
    }
}
