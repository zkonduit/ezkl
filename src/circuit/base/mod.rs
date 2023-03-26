/// Layouts for specific functions (composed of base ops)
pub mod layouts;

/// Tests
#[cfg(test)]
mod tests;

use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use halo2curves::FieldExt;
use itertools::Itertools;
use log::trace;
use serde::{Deserialize, Serialize};

use crate::tensor::{self, Tensor, TensorError, TensorType, ValTensor, VarTensor};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BaseOp {
    Dot,
    Identity,
    Add,
    Mult,
    Sub,
    Sum,
}

#[allow(missing_docs)]
/// An enum representing activating the sanity checks we can perform on the accumulated arguments
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CheckMode {
    SAFE,
    UNSAFE,
}

impl From<String> for CheckMode {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "safe" => CheckMode::SAFE,
            "unsafe" => CheckMode::UNSAFE,
            _ => panic!("not a valid checkmode"),
        }
    }
}

/// Matches a [BaseOp] to an operation over inputs
impl BaseOp {
    /// forward func
    pub fn f<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        &self,
        inputs: (T, T, T),
    ) -> T {
        let (a, b, m) = inputs;
        match &self {
            BaseOp::Dot => a * b + m,
            BaseOp::Add => a + b,
            BaseOp::Identity => b,
            BaseOp::Sum => b + m,
            BaseOp::Sub => a - b,
            BaseOp::Mult => a * b,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::Identity => "IDENTITY",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
            BaseOp::Sub => "SUB",
            BaseOp::Mult => "MULT",
            BaseOp::Sum => "SUM",
        }
    }
    fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::Identity => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::Add => (0, 1),
            BaseOp::Sub => (0, 1),
            BaseOp::Mult => (0, 1),
            BaseOp::Sum => (-1, 2),
        }
    }
    fn num_inputs(&self) -> usize {
        match self {
            BaseOp::Identity => 1,
            BaseOp::Dot => 2,
            BaseOp::Add => 2,
            BaseOp::Sub => 2,
            BaseOp::Mult => 2,
            BaseOp::Sum => 1,
        }
    }
    fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::Identity => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 0,
            BaseOp::Sub => 0,
            BaseOp::Mult => 0,
            BaseOp::Sum => 1,
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Dot,
    Matmul,
    Affine,
    Conv {
        padding: (usize, usize),
        stride: (usize, usize),
    },
    SumPool {
        padding: (usize, usize),
        stride: (usize, usize),
        kernel_shape: (usize, usize),
    },
    Add,
    Sub,
    Mult,
    Identity,
    Reshape(Vec<usize>),
    Flatten(Vec<usize>),
    BatchNorm,
    ScaleAndShift,
    Pad(usize, usize),
    Sum,
    Pow(u32),
    Pack(u32, u32),
    GlobalSumPool,
    Rescaled {
        inner: Box<Op>,
        scale: Vec<(usize, usize)>,
    },
}

impl Op {
    /// circuit shape
    pub fn circuit_shapes(&self, input_shapes: Vec<Vec<usize>>) -> Vec<usize> {
        let mut shapes = match &self {
            Op::Identity => vec![0, input_shapes[0].iter().product()],
            Op::Reshape(_) => vec![],
            Op::Flatten(_) => vec![],
            Op::Pad(_, _) => vec![],
            Op::Add => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Mult => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Sub => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Sum => vec![0, input_shapes[0].iter().product()],
            Op::Dot => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Pow(_) => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Pack(_, _) => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::GlobalSumPool => unreachable!("should be handled by sumpool"),
            Op::ScaleAndShift => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::BatchNorm => input_shapes.iter().map(|x| x.iter().product()).collect(),
            Op::Conv { padding, stride } => {
                let image_dims = &input_shapes[0];
                let kernel_dims = &input_shapes[1];

                let (output_channels, _input_channels, kernel_height, kernel_width) = (
                    kernel_dims[0],
                    kernel_dims[1],
                    kernel_dims[2],
                    kernel_dims[3],
                );

                let (image_height, image_width) = (image_dims[1], image_dims[2]);

                let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
                let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

                let output_len = output_channels
                    * vert_slides
                    * horz_slides
                    * image_dims.iter().product::<usize>();

                vec![output_len; 2]
            }
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => {
                let image_dims = &input_shapes[0];

                let (image_height, image_width) = (image_dims[1], image_dims[2]);

                let vert_slides = (image_height + 2 * padding.0 - kernel_shape.0) / stride.0 + 1;
                let horz_slides = (image_width + 2 * padding.1 - kernel_shape.1) / stride.1 + 1;

                let output_len = image_dims[0]
                    * vert_slides
                    * horz_slides
                    // add 1 cause of bias
                    * (image_dims.iter().product::<usize>() + 1);

                vec![output_len; 2]
            }
            Op::Affine => {
                let s = input_shapes.clone();
                // add 1 cause of bias
                let output_len = s[0][0] * (s[0][1] + 1) * s[1][1];
                vec![output_len; 3]
            }
            Op::Matmul => {
                let s = input_shapes.clone();
                let output_len = s[0].iter().product::<usize>() * s[1][1];
                vec![output_len; 3]
            }
            Op::Rescaled { inner, .. } => inner.circuit_shapes(input_shapes),
        };
        match shapes.last() {
            // add output
            Some(s) => shapes.push(s.clone()),
            _ => {}
        };
        shapes
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub fn f<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        &self,
        mut inputs: Vec<Tensor<T>>,
    ) -> Result<Tensor<T>, TensorError> {
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
            Op::Pad(dim1, dim2) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pad inputs".to_string()));
                }
                tensor::ops::pad(&inputs[0], (*dim1, *dim2))
            }
            Op::Add => tensor::ops::add(&inputs),
            Op::Sub => tensor::ops::sub(&inputs),
            Op::Mult => tensor::ops::mult(&inputs),
            Op::Affine => tensor::ops::affine(&inputs),
            Op::BatchNorm => tensor::ops::scale_and_shift(&inputs),
            Op::ScaleAndShift => tensor::ops::scale_and_shift(&inputs),
            Op::Matmul => tensor::ops::matmul(&inputs),
            Op::Dot => tensor::ops::dot(&inputs.iter().collect()),
            Op::Conv { padding, stride } => tensor::ops::convolution(&inputs, *padding, *stride),
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => tensor::ops::sumpool(&inputs[0], *padding, *stride, *kernel_shape),
            Op::Pack(base, scale) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pack inputs".to_string()));
                }
                // these unwraps should never ever fail if the Tensortypes are correctly implemented
                // if anything we want these to hard fail if not implemented
                let mut base_t = T::zero().unwrap();
                for _ in 0..*base {
                    base_t = base_t + T::one().unwrap();
                }
                tensor::ops::pack(&inputs[0], base_t, *scale)
            }
            Op::Pow(u) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pow inputs".to_string()));
                }
                inputs[0].pow(*u)
            }
            Op::Sum => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("sum inputs".to_string()));
                }
                tensor::ops::sum(&inputs[0])
            }
            Op::Rescaled { inner, scale } => {
                if scale.len() != inputs.len() {
                    return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
                }

                let mut rescaled_inputs = vec![];
                for (i, ri) in inputs.iter_mut().enumerate() {
                    rescaled_inputs.push(tensor::ops::rescale(ri, scale[i].1)?);
                }
                Ok(inner.f(rescaled_inputs)?)
            }
            Op::GlobalSumPool => unreachable!(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Identity => write!(f, "identity"),
            Op::Reshape(new_dims) => write!(f, "reshape to {:?}", new_dims),
            Op::Flatten(new_dims) => write!(f, "flatten to {:?}", new_dims),
            Op::Pad(dim1, dim2) => write!(f, "padding: ({:?}, {:?})", dim1, dim2),
            Op::Add => write!(f, "add"),
            Op::Sub => write!(f, "sub"),
            Op::Sum => write!(f, "sum"),
            Op::Mult => write!(f, "mult"),
            Op::Matmul => write!(f, "matmul"),
            Op::Dot => write!(f, "dot"),
            Op::Pack(base, _) => write!(f, "pack with base {:?}", base),
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

/// Configuration for an accumulated arg.
#[derive(Clone, Debug)]
pub struct BaseConfig<F: FieldExt + TensorType> {
    /// the inputs to the accumulated operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the accumulated operations.
    pub output: VarTensor,
    /// [Selectors] generated when configuring the layer. We use a BTreeMap as we expect to configure many base gates.
    pub selectors: BTreeMap<(BaseOp, usize), Selector>,
    /// Activate sanity checks
    pub check_mode: CheckMode,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> BaseConfig<F> {
    /// Configures [BaseOp]s for a given [ConstraintSystem].
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations.
    /// * `output` - The variable representing the (currently singular) output of the operations.
    /// * `check_mode` - The variable representing the (currently singular) output of the operations.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor; 2],
        output: &VarTensor,
        check_mode: CheckMode,
    ) -> Self {
        // setup a selector per base op
        let mut selectors = BTreeMap::new();

        assert!(inputs[0].num_cols() == inputs[1].num_cols());
        assert!(inputs[0].num_cols() == output.num_cols());

        for i in 0..output.num_cols() {
            selectors.insert((BaseOp::Add, i), meta.selector());
            selectors.insert((BaseOp::Sub, i), meta.selector());
            selectors.insert((BaseOp::Dot, i), meta.selector());
            selectors.insert((BaseOp::Sum, i), meta.selector());
            selectors.insert((BaseOp::Mult, i), meta.selector());
            selectors.insert((BaseOp::Identity, i), meta.selector());
        }

        let config = Self {
            selectors,
            inputs: inputs.to_vec(),
            output: output.clone(),
            check_mode,
            _marker: PhantomData,
        };

        for ((base_op, col_idx), selector) in config.selectors.iter() {
            meta.create_gate(base_op.as_str(), |meta| {
                let selector = meta.query_selector(*selector);
                let idx_offset = col_idx * output.col_size();
                let mut qis = vec![Expression::<F>::zero().unwrap(); 2];
                for (i, q_i) in qis
                    .iter_mut()
                    .enumerate()
                    .take(2)
                    .skip(2 - base_op.num_inputs())
                {
                    *q_i = config.inputs[i]
                        .query_rng(meta, 0, idx_offset, 1)
                        .expect("accum: input query failed")[0]
                        .clone()
                }

                // Get output expressions for each input channel
                let (rotation_offset, rng) = base_op.query_offset_rng();

                let expected_output: Tensor<Expression<F>> = config
                    .output
                    .query_rng(meta, rotation_offset, idx_offset, rng)
                    .expect("poly: output query failed");

                let res = base_op.f((qis[0].clone(), qis[1].clone(), expected_output[0].clone()));

                let constraints = vec![expected_output[base_op.constraint_idx()].clone() - res];

                Constraints::with_selector(selector, constraints)
            });
        }

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    /// * `offset` - Offset to assign.
    /// * `op` - The operation being represented.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
        offset: usize,
        op: Op,
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        let mut cp_values = vec![];
        for v in values.iter() {
            if let ValTensor::Instance { .. } = v {
                cp_values.push(layouts::identity(self, layouter, &[v.clone()], offset)?);
            } else {
                cp_values.push(v.clone());
            }
        }
        trace!("laying out {}", op);
        match op {
            Op::Dot => layouts::dot(self, layouter, cp_values[..].try_into()?, offset),
            Op::Sum => layouts::sum(self, layouter, cp_values[..].try_into()?, offset),
            Op::Matmul => layouts::matmul(self, layouter, cp_values[..].try_into()?, offset),
            Op::Affine => layouts::affine(self, layouter, cp_values[..].try_into()?, offset),
            Op::Conv { padding, stride } => layouts::conv(
                self,
                layouter,
                cp_values[..].try_into()?,
                padding,
                stride,
                offset,
            ),
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => layouts::sumpool(
                self,
                layouter,
                cp_values[..].try_into()?,
                padding,
                stride,
                kernel_shape,
                offset,
            ),
            Op::Add => layouts::pairwise(
                self,
                layouter,
                cp_values[..].try_into()?,
                offset,
                BaseOp::Add,
            ),
            Op::Sub => layouts::pairwise(
                self,
                layouter,
                cp_values[..].try_into()?,
                offset,
                BaseOp::Sub,
            ),
            Op::Mult => layouts::pairwise(
                self,
                layouter,
                cp_values[..].try_into()?,
                offset,
                BaseOp::Mult,
            ),
            Op::Identity => layouts::identity(self, layouter, cp_values[..].try_into()?, offset),
            Op::Reshape(d) | Op::Flatten(d) => layouts::reshape(cp_values[..].try_into()?, &d),
            Op::BatchNorm => {
                layouts::scale_and_shift(self, layouter, cp_values[..].try_into()?, offset)
            }
            Op::ScaleAndShift => {
                layouts::scale_and_shift(self, layouter, cp_values[..].try_into()?, offset)
            }
            Op::Pad(p1, p2) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError));
                }
                let mut input = cp_values[0].clone();
                input.pad((p1, p2))?;
                Ok(input)
            }
            Op::Pow(exp) => layouts::pow(self, layouter, cp_values[..].try_into()?, exp, offset),
            Op::Pack(base, scale) => layouts::pack(
                self,
                layouter,
                cp_values[..].try_into()?,
                base,
                scale,
                offset,
            ),
            Op::Rescaled { inner, scale } => {
                if scale.len() != values.len() {
                    return Err(Box::new(TensorError::DimMismatch(
                        "rescaled inputs".to_string(),
                    )));
                }

                let res = &layouts::rescale(
                    self,
                    layouter,
                    cp_values[..].try_into()?,
                    &scale,
                    offset,
                )?[..];
                self.layout(layouter, res, offset, *inner)
            }
            Op::GlobalSumPool => unreachable!(),
        }
    }
}
