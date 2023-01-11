use super::*;
use crate::abort;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use itertools::Itertools;
use log::error;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

#[allow(missing_docs)]
/// An enum representing the operations that can be merged into a single circuit gate.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Identity,
    Reshape(Vec<usize>),
    Flatten(Vec<usize>),
    Add,
    Sub,
    Sum,
    Mult,
    Matmul,
    Dot,
    Affine,
    BatchNorm,
    ScaleAndShift,
    Conv {
        padding: (usize, usize),
        stride: (usize, usize),
    },
    SumPool {
        padding: (usize, usize),
        stride: (usize, usize),
        kernel_shape: (usize, usize),
    },
    GlobalSumPool,
    Pow(usize),
    Rescaled {
        inner: Box<Op>,
        scale: Vec<(usize, usize)>,
    },
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Identity => write!(f, "identity"),
            Op::Reshape(new_dims) => write!(f, "reshape to {:?}", new_dims),
            Op::Flatten(new_dims) => write!(f, "flatten to {:?}", new_dims),
            Op::Add => write!(f, "add"),
            Op::Sub => write!(f, "sub"),
            Op::Sum => write!(f, "sum"),
            Op::Mult => write!(f, "mult"),
            Op::Matmul => write!(f, "matmul"),
            Op::Dot => write!(f, "dot"),
            Op::Affine => write!(f, "affine"),
            Op::BatchNorm => write!(f, "batchnorm"),
            Op::ScaleAndShift => write!(f, "scale & shift"),
            Op::Conv { padding, stride } => {
                write!(f, "conv w/ padding: {:?}, stride: {:?}", padding, stride)
            }
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => {
                write!(
                    f,
                    "avg pl w/ padding: {:?}, stride: {:?}, kernel shape: {:?}",
                    padding, stride, kernel_shape,
                )
            }
            Op::GlobalSumPool => write!(f, "globalsumpool"),
            Op::Pow(s) => write!(f, "pow {}", s),
            Op::Rescaled { inner, scale } => {
                write!(
                    f,
                    "{} w/ scalings: {:?}",
                    **inner,
                    scale.iter().map(|e| e.1).collect_vec()
                )
            }
        }
    }
}

impl Op {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub fn f<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        &self,
        mut inputs: Vec<Tensor<T>>,
    ) -> Result<Tensor<T>, Box<dyn Error>> {
        match &self {
            Op::Identity => Ok(inputs[0].clone()),
            Op::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            Op::Flatten(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            Op::Add => add(&inputs),
            Op::Sub => sub(&inputs),
            Op::Mult => mult(&inputs),
            Op::Affine => affine(&inputs),
            Op::BatchNorm => scale_and_shift(&inputs),
            Op::ScaleAndShift => scale_and_shift(&inputs),
            Op::Matmul => matmul(&inputs),
            Op::Dot => {
                todo!();
            }
            Op::Conv { padding, stride } => convolution(&inputs, *padding, *stride),
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => sumpool(&inputs[0], *padding, *stride, *kernel_shape),
            Op::GlobalSumPool => unreachable!(),
            Op::Pow(u) => {
                if 1 != inputs.len() {
                    return Err(Box::new(CircuitError::DimMismatch(
                        "pow constraint".to_string(),
                    )));
                }
                pow(&inputs[0], *u)
            }
            Op::Sum => {
                if 1 != inputs.len() {
                    return Err(Box::new(CircuitError::DimMismatch(
                        "sum constraint".to_string(),
                    )));
                }
                sum(&inputs[0])
            }
            Op::Rescaled { inner, scale } => {
                if scale.len() != inputs.len() {
                    return Err(Box::new(CircuitError::DimMismatch(
                        "rescaled constraint".to_string(),
                    )));
                }

                let mut rescaled_inputs = vec![];
                for (i, ri) in inputs.iter_mut().enumerate() {
                    rescaled_inputs.push(rescale(ri, scale[i].1)?);
                }
                Ok(inner.f(rescaled_inputs)?)
            }
        }
    }
}

/// Representation of a the inputs a [Node] can ingest. The inner type indexes over each of the types.
#[derive(Clone, Debug)]
pub enum InputType {
    /// an explicit input to the operations
    Input(usize),
    /// the intermediate output of a [Node]
    Inter(usize),
}

/// Representation of a single fuseable operation.
#[derive(Clone, Debug)]
pub struct Node {
    /// the type of operation
    pub op: Op,
    /// execution order over explicit inputs and intermediate outputs.
    pub input_order: Vec<InputType>,
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct Config<F: FieldExt + TensorType> {
    /// the inputs to the fused operations.
    pub inputs: Vec<VarTensor>,
    /// the set of [Node] represented in the operation.
    nodes: Vec<Node>,
    /// the (currently singular) output of the fused operations.
    pub output: VarTensor,
    /// [Selector] generated when configuring the layer.
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures the sequence of operations into a circuit gate, represented as an array of [Node].
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations. [Node]s index over these inputs using their `input_order` attribute. They can also index over the intermediate outputs of other [Node]s.
    /// * `output` - The variable representing the (currently singular) output of the fused operations.
    /// * `nodes` - The sequence of operations (in order of execution) that constitute the fused operation.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor],
        output: &VarTensor,
        nodes: &[Node],
    ) -> Self {
        let mut config = Self {
            selector: meta.selector(),
            nodes: nodes.to_vec(),
            inputs: inputs.to_vec(),
            output: output.clone(),
            _marker: PhantomData,
        };

        meta.create_gate("basic_op", |meta| {
            let selector = meta.query_selector(config.selector);
            let qis = config
                .inputs
                .iter()
                .map(|input| match input.query(meta, 0) {
                    Ok(q) => q,
                    Err(e) => {
                        abort!("failed to query input {:?}", e);
                    }
                })
                .collect::<Vec<_>>();

            let mut config_outputs = vec![];
            for node in config.nodes.iter_mut() {
                match Self::apply_op(node, &qis, &mut config_outputs) {
                    Ok(res) => res,
                    e => {
                        abort!("apply op failed {:?}", e);
                    }
                };
            }
            let witnessed_output = &config_outputs[config.nodes.len() - 1];

            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = match config.output.query(meta, 0) {
                Ok(res) => res,
                Err(e) => {
                    abort!("failed to query output during fused layer layout {:?}", e);
                }
            };

            let constraints = witnessed_output
                .enum_map(|i, o| o - expected_output[i].clone())
                .unwrap();

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations. [Node]s index over these inputs using their `input_order` attribute. They can also index over the intermediate outputs of other [Node]s.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        if values.len() != self.inputs.len() {
            return Err(Box::new(CircuitError::DimMismatch(
                "polynomial layout".to_string(),
            )));
        }

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let offset = 0;
                self.selector.enable(&mut region, offset)?;

                let mut inputs = vec![];
                for (i, input) in values.iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            match self.inputs[i].assign(&mut region, offset, input) {
                                Ok(res) => res.map(|e| e.value_field().evaluate()),
                                Err(e) => {
                                    abort!(
                                        "failed to assign inputs during fused layer layout {:?}",
                                        e
                                    );
                                }
                            }
                        },
                        input,
                    );
                    inputs.push(inp);
                }

                let mut layout_outputs = vec![];

                for node in self.nodes.iter_mut() {
                    match Self::apply_op(node, &inputs, &mut layout_outputs) {
                        Ok(res) => res,
                        Err(e) => {
                            abort!("apply op failed {:?}", e);
                        }
                    };
                }
                let output: ValTensor<F> = match layout_outputs.last() {
                    Some(a) => a.clone().into(),
                    None => {
                        panic!("fused layer has empty outputs");
                    }
                };

                match self.output.assign(&mut region, offset, &output) {
                    Ok(a) => Ok(a),
                    Err(e) => {
                        abort!("failed to assign fused layer output {:?}", e);
                    }
                }
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                return Err(Box::new(e));
            }
        };

        Ok(ValTensor::from(t))
    }

    /// Applies an operation represented by a [Op] to the set of inputs (both explicit and intermediate results) it indexes over.
    pub fn apply_op<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        node: &mut Node,
        inputs: &[Tensor<T>],
        outputs: &mut Vec<Tensor<T>>,
    ) -> Result<(), Box<dyn Error>> {
        let op_inputs = node
            .input_order
            .iter()
            .map(|input| match input {
                InputType::Input(u) => inputs[*u].clone(),
                InputType::Inter(u) => outputs[*u].clone(),
            })
            .collect_vec();
        outputs.push(node.op.f(op_inputs)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        arithmetic::{Field, FieldExt},
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    use rand::rngs::OsRng;

    const K: usize = 4;
    const LEN: usize = 2;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        input: ValTensor<F>,
        l0_params: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = Config<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let input = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let kernel = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
            let bias = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            // tells the config layer to add an affine op to a circuit gate
            let affine_node = Node {
                op: Op::Affine,
                input_order: vec![
                    InputType::Input(0),
                    InputType::Input(1),
                    InputType::Input(2),
                ],
            };

            Self::Config::configure(cs, &[input, kernel, bias], &output, &[affine_node])
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config.layout(
                &mut layouter,
                &[
                    self.input.clone(),
                    self.l0_params[0].clone(),
                    self.l0_params[1].clone(),
                ],
            );
            Ok(())
        }
    }

    #[test]
    fn affinecircuit() {
        // parameters
        let mut l0_kernel =
            Tensor::from((0..LEN * LEN).map(|_| Value::known(pallas::Base::random(OsRng))));
        l0_kernel.reshape(&[LEN, LEN]);

        let l0_bias = Tensor::from((0..LEN).map(|_| Value::known(pallas::Base::random(OsRng))));

        let input = Tensor::from((0..LEN).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = MyCircuit::<F> {
            input: ValTensor::from(input),
            l0_params: [ValTensor::from(l0_kernel), ValTensor::from(l0_bias)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
