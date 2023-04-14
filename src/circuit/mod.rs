/// Layouts for specific functions (composed of base ops)
pub mod layouts;

///
pub mod table;

///
pub mod utils;

/// Tests
#[cfg(test)]
mod tests;

use thiserror::Error;

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Region},
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use halo2curves::FieldExt;
use itertools::Itertools;
use log::{trace, warn};
use serde::{Deserialize, Serialize};

use crate::{
    fieldutils::{i128_to_felt, i32_to_felt},
    tensor::{self, Tensor, TensorError, TensorType, ValTensor, VarTensor},
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    error::Error,
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use self::table::Table;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Shape mismatch in circuit construction
    #[error("dimension mismatch in circuit construction for op: {0}")]
    DimMismatch(String),
    /// Error when instantiating lookup tables
    #[error("failed to instantiate lookup tables")]
    LookupInstantiation,
    /// A lookup table was was already assigned
    #[error("attempting to initialize an already instantiated lookup table")]
    TableAlreadyAssigned,
    /// This operation is unsupported
    #[error("unsupported operation in graph")]
    UnsupportedOp,
}

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
    Neg,
    Range { tol: i32 },
}

#[allow(missing_docs)]
/// An enum representing activating the sanity checks we can perform on the accumulated arguments
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default, Copy,
)]
pub enum CheckMode {
    #[default]
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
    pub fn f<
        T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    >(
        &self,
        inputs: (T, T, T),
    ) -> T {
        let (a, b, m) = inputs;
        match &self {
            BaseOp::Dot => a * b + m,
            BaseOp::Add => a + b,
            BaseOp::Identity => b,
            BaseOp::Sum => b + m,
            BaseOp::Neg => -b,
            BaseOp::Sub => a - b,
            BaseOp::Mult => a * b,
            BaseOp::Range { .. } => b,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::Identity => "IDENTITY",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
            BaseOp::Neg => "NEG",
            BaseOp::Sub => "SUB",
            BaseOp::Mult => "MULT",
            BaseOp::Sum => "SUM",
            BaseOp::Range { .. } => "RANGE",
        }
    }
    fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::Identity => (0, 1),
            BaseOp::Neg => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::Add => (0, 1),
            BaseOp::Sub => (0, 1),
            BaseOp::Mult => (0, 1),
            BaseOp::Sum => (-1, 2),
            BaseOp::Range { .. } => (0, 1),
        }
    }
    fn num_inputs(&self) -> usize {
        match self {
            BaseOp::Identity => 1,
            BaseOp::Neg => 1,
            BaseOp::Dot => 2,
            BaseOp::Add => 2,
            BaseOp::Sub => 2,
            BaseOp::Mult => 2,
            BaseOp::Sum => 1,
            BaseOp::Range { .. } => 1,
        }
    }
    fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::Identity => 0,
            BaseOp::Neg => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 0,
            BaseOp::Sub => 0,
            BaseOp::Mult => 0,
            BaseOp::Range { .. } => 0,
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Div {
        denom: utils::F32,
    },
    ReLU {
        scale: usize,
    },
    Sqrt {
        scales: (usize, usize),
    },
    LeakyReLU {
        scale: usize,
        slope: utils::F32,
    },
    PReLU {
        scale: usize,
        slopes: Vec<utils::F32>,
    },
    Sigmoid {
        scales: (usize, usize),
    },
    Tanh {
        scales: (usize, usize),
    },
    Erf {
        scales: (usize, usize),
    },
    Mean {
        scale: usize,
    },
}

impl LookupOp {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub fn f(&self, x: Tensor<i128>) -> Result<Tensor<i128>, TensorError> {
        match &self {
            LookupOp::Div { denom } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*denom),
            )),
            LookupOp::ReLU { scale } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, *scale, 0_f32))
            }
            LookupOp::LeakyReLU { scale, slope } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, *scale, slope.0))
            }
            LookupOp::PReLU { scale, slopes } => Ok(tensor::ops::nonlinearities::prelu(
                &x,
                *scale,
                &slopes.iter().map(|e| e.0).collect_vec(),
            )),
            LookupOp::Sigmoid { scales } => {
                Ok(tensor::ops::nonlinearities::sigmoid(&x, scales.0, scales.1))
            }
            LookupOp::Sqrt { scales } => {
                Ok(tensor::ops::nonlinearities::sqrt(&x, scales.0, scales.1))
            }
            LookupOp::Tanh { scales } => {
                Ok(tensor::ops::nonlinearities::tanh(&x, scales.0, scales.1))
            }
            LookupOp::Erf { scales } => {
                Ok(tensor::ops::nonlinearities::erffunc(&x, scales.0, scales.1))
            }
            LookupOp::Mean { scale } => Ok(tensor::ops::nonlinearities::mean(&x, *scale)),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            LookupOp::Div { .. } => "DIV",
            LookupOp::ReLU { .. } => "RELU",
            LookupOp::LeakyReLU { .. } => "LEAKY_RELU",
            LookupOp::PReLU { .. } => "PRELU",
            LookupOp::Sigmoid { .. } => "SIGMOID",
            LookupOp::Sqrt { .. } => "SQRT",
            LookupOp::Tanh { .. } => "TANH",
            LookupOp::Erf { .. } => "ERF",
            LookupOp::Mean { .. } => "MEAN",
        }
    }

    /// a value which is always in the table
    pub fn default_pair<F: FieldExt>(&self) -> (F, F) {
        let x = vec![0_i128].into_iter().into();
        (F::zero(), i128_to_felt(self.f(x).unwrap()[0]))
    }
}

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
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
    RangeCheck(i32),
}

impl Op {
    /// circuit shape
    pub fn circuit_shapes(&self, input_shapes: Vec<Vec<usize>>) -> Vec<usize> {
        let mut shapes = match &self {
            Op::Identity => vec![0, input_shapes[0].iter().product()],
            Op::Reshape(_) => vec![0; 2],
            Op::Flatten(_) => vec![0; 2],
            Op::Pad(_, _) => vec![0; 2],
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

                let padded_height = image_height + 2 * padding.0;
                let padded_width = image_width + 2 * padding.1;

                let vert_slides = (padded_height - kernel_height) / stride.0 + 1;
                let horz_slides = (padded_width - kernel_width) / stride.1 + 1;

                let input_shapes = vec![
                    vec![
                        output_channels * vert_slides * horz_slides,
                        (padded_height * padded_width * image_dims[0] + 1),
                    ],
                    vec![(padded_height * padded_width * image_dims[0] + 1), 1],
                ];
                let op = Op::Matmul;
                let output_len = op.circuit_shapes(input_shapes);

                vec![*output_len.last().unwrap(); 2]
            }
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => {
                let image_dims = &input_shapes[0];

                let (image_height, image_width) = (image_dims[1], image_dims[2]);

                let padded_height = image_height + 2 * padding.0;
                let padded_width = image_width + 2 * padding.1;

                let vert_slides = (padded_height - kernel_shape.0) / stride.0 + 1;
                let horz_slides = (padded_width - kernel_shape.1) / stride.1 + 1;

                let input_shapes = vec![
                    vec![
                        image_dims[0] * vert_slides * horz_slides,
                        (padded_height * padded_width * image_dims[0] + 1),
                    ],
                    vec![(padded_height * padded_width * image_dims[0] + 1), 1],
                ];
                let op = Op::Matmul;
                let output_len = op.circuit_shapes(input_shapes);

                vec![*output_len.last().unwrap(); 2]
            }
            Op::Affine => {
                let s = input_shapes;
                // add 1 cause of bias
                let output_len = s[1][0] * (s[1][1] + 1);
                vec![output_len; 2]
            }
            Op::Matmul => {
                let output_len = input_shapes[0].iter().product::<usize>() * input_shapes[1][1];

                vec![output_len; 2]
            }
            Op::Rescaled { inner, .. } => inner.circuit_shapes(input_shapes),
            Op::RangeCheck(..) => input_shapes.iter().map(|x| x.iter().product()).collect(),
        };
        match shapes.last() {
            // add output
            Some(s) => shapes.push(*s),
            _ => {}
        };
        shapes
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub fn f(&self, mut inputs: Vec<Tensor<i128>>) -> Result<Tensor<i128>, TensorError> {
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

                tensor::ops::pack(&inputs[0], *base as i128, *scale)
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
            Op::RangeCheck(..) => Ok(inputs[0].clone()),
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
                    "rescaled {} w/ scalings: {:?}",
                    **inner,
                    scale.iter().map(|e| e.1).collect_vec()
                )
            }
            Op::RangeCheck(tol) => write!(f, "range check w/ tol {}", tol),
        }
    }
}

// Initially, some of these OpKinds will be folded into others (for example, Const nodes that
// contain parameters will be handled at the consuming self.
// Eventually, though, we probably want to keep them and treat them directly (layouting and configuring
// at each type of node)
/// Enum of the different kinds of operations `ezkl` can support.
#[derive(Clone, Debug, Default, PartialEq, Eq, Ord, PartialOrd, Deserialize, Serialize)]
pub enum OpKind {
    /// A nonlinearity
    Lookup(LookupOp),
    /// A fused op, combining affine layers or other arithmetic
    Poly(Op),
    /// Constant
    Const,
    /// Input node
    Input,
    /// Unable to parse the node type
    Unknown(String),
    #[allow(missing_docs)]
    #[default]
    None,
}

impl From<Op> for OpKind {
    fn from(op: Op) -> Self {
        OpKind::Poly(op)
    }
}

impl From<LookupOp> for OpKind {
    fn from(op: LookupOp) -> Self {
        OpKind::Lookup(op)
    }
}

impl OpKind {
    /// Produce an OpKind from a `&str` onnx name  
    pub fn new(name: &str) -> Self {
        match name {
            "Clip" => OpKind::Lookup(LookupOp::ReLU { scale: 1 }),
            "Prelu" => OpKind::Lookup(LookupOp::PReLU {
                scale: 1,
                slopes: vec![],
            }),
            "LeakyRelu" => OpKind::Lookup(LookupOp::LeakyReLU {
                scale: 1,
                slope: utils::F32(0.0),
            }),
            "Sigmoid" => OpKind::Lookup(LookupOp::Sigmoid { scales: (1, 1) }),
            "Sqrt" => OpKind::Lookup(LookupOp::Sqrt { scales: (1, 1) }),
            "Tanh" => OpKind::Lookup(LookupOp::Tanh { scales: (1, 1) }),
            "onnx.Erf" => OpKind::Lookup(LookupOp::Erf { scales: (1, 1) }),
            "Div" => OpKind::Lookup(LookupOp::Div {
                denom: utils::F32(1.0),
            }),
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "Add" => OpKind::Poly(Op::Add),
            "Sub" => OpKind::Poly(Op::Sub),
            "Mul" => OpKind::Poly(Op::Mult),
            "Gemm" => OpKind::Poly(Op::Affine),
            "MatMulInference" => OpKind::Poly(Op::Matmul),
            "Dot" => OpKind::Poly(Op::Dot),
            "Reduce<Sum>" => OpKind::Poly(Op::Sum),
            "Reduce<Mean>" => OpKind::Lookup(LookupOp::Mean { scale: 1 }),
            "Pow" => OpKind::Poly(Op::Pow(1)),
            "Conv" | "ConvHir" => OpKind::Poly(Op::Conv {
                padding: (1, 1),
                stride: (1, 1),
            }),
            "SumPool" => OpKind::Poly(Op::SumPool {
                padding: (1, 1),
                stride: (1, 1),
                kernel_shape: (1, 1),
            }),
            "GlobalAvgPool" => OpKind::Poly(Op::GlobalSumPool),
            "Pad" => OpKind::Poly(Op::Pad(0, 0)),
            "Reshape" => OpKind::Poly(Op::Reshape(Vec::new())),
            "Flatten" => OpKind::Poly(Op::Flatten(Vec::new())),
            "BatchNorm" => OpKind::Poly(Op::BatchNorm),
            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown(c.to_string())
            }
        }
    }
    /// is ploy type constrant
    pub fn is_poly(&self) -> bool {
        matches!(self, OpKind::Poly(_))
    }

    /// is lookup based op
    pub fn is_lookup(&self) -> bool {
        matches!(self, OpKind::Lookup(_))
    }

    /// is lookup based op
    pub fn is_parameterized(&self) -> bool {
        match self {
            OpKind::Poly(Op::Affine) | OpKind::Poly(Op::Conv { .. }) => true,
            _ => false,
        }
    }

    /// is rescaled op
    pub fn is_rescaled(&self) -> bool {
        matches!(self, OpKind::Poly(Op::Rescaled { .. }))
    }

    /// is input
    pub fn is_input(&self) -> bool {
        matches!(self, OpKind::Input)
    }

    /// is const
    pub fn is_const(&self) -> bool {
        matches!(self, OpKind::Const)
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Const => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Lookup(s) => write!(f, "{:#?}", s),
            OpKind::Poly(s) => write!(f, "{}", s),
            OpKind::Unknown(c) => write!(f, "? {}", c),
            OpKind::None => write!(f, "n/a",),
        }
    }
}

/// Configuration for an accumulated arg.
#[derive(Clone, Debug, Default)]
pub struct BaseConfig<F: FieldExt + TensorType> {
    /// the inputs to the accumulated operations.
    pub inputs: Vec<VarTensor>,
    /// the VarTensor reserved for lookup operations (could be an element of inputs)
    /// Note that you should be careful to ensure that the lookup_input is not simultaneously assigned to by other non-lookup operations eg. in the case of composite ops.
    pub lookup_input: VarTensor,
    /// the (currently singular) output of the accumulated operations.
    pub output: VarTensor,
    ///  the VarTensor reserved for lookup operations (could be an element of inputs or the same as output)
    /// Note that you should be careful to ensure that the lookup_output is not simultaneously assigned to by other non-lookup operations eg. in the case of composite ops.
    pub lookup_output: VarTensor,
    /// [Selectors] generated when configuring the layer. We use a BTreeMap as we expect to configure many base gates.
    pub selectors: BTreeMap<(BaseOp, usize), Selector>,
    /// [Selectors] generated when configuring the layer. We use a BTreeMap as we expect to configure many lookup ops.
    pub lookup_selectors: BTreeMap<(LookupOp, usize), Selector>,
    /// [Table]
    pub tables: BTreeMap<LookupOp, Rc<RefCell<Table<F>>>>,
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
        tol: i32,
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
            selectors.insert((BaseOp::Range { tol }, i), meta.selector());
        }

        // Given a range R and a value v, returns the expression
        // (v) * (1 - v) * (2 - v) * ... * (R - 1 - v)
        let range_check = |tol: i32, value: Expression<F>| {
            (-tol..tol).fold(value.clone(), |expr, i| {
                expr * (Expression::Constant(i32_to_felt(i)) - value.clone())
            })
        };

        for ((base_op, col_idx), selector) in selectors.iter() {
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
                    *q_i = inputs[i]
                        .query_rng(meta, 0, idx_offset, 1)
                        .expect("accum: input query failed")[0]
                        .clone()
                }

                // Get output expressions for each input channel
                let (rotation_offset, rng) = base_op.query_offset_rng();

                let expected_output: Tensor<Expression<F>> = output
                    .query_rng(meta, rotation_offset, idx_offset, rng)
                    .expect("poly: output query failed");

                let res = base_op.f((qis[0].clone(), qis[1].clone(), expected_output[0].clone()));

                let constraints = match base_op {
                    BaseOp::Range { tol } => {
                        vec![range_check(
                            *tol,
                            res - expected_output[base_op.constraint_idx()].clone(),
                        )]
                    }
                    _ => vec![expected_output[base_op.constraint_idx()].clone() - res],
                };

                Constraints::with_selector(selector, constraints)
            });
        }

        Self {
            selectors,
            lookup_selectors: BTreeMap::new(),
            inputs: inputs.to_vec(),
            lookup_input: VarTensor::None,
            lookup_output: VarTensor::None,
            tables: BTreeMap::new(),
            output: output.clone(),
            check_mode,
            _marker: PhantomData,
        }
    }

    /// Configures and creates lookup selectors
    pub fn configure_lookup(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        bits: usize,
        nl: &LookupOp,
    ) -> Result<(), Box<dyn Error>> {
        let mut selectors = BTreeMap::new();
        let table =
            if let std::collections::btree_map::Entry::Vacant(e) = self.tables.entry(nl.clone()) {
                let table = Rc::new(RefCell::new(Table::<F>::configure(cs, bits, nl)));
                e.insert(table.clone());
                table
            } else {
                return Ok(());
            };
        for x in 0..input.num_cols() {
            let qlookup = cs.complex_selector();
            selectors.insert((nl.clone(), x), qlookup);
            let _ = cs.lookup(nl.as_str(), |cs| {
                let qlookup = cs.query_selector(qlookup);
                let not_qlookup = Expression::Constant(<F as Field>::one()) - qlookup.clone();
                let (default_x, default_y): (F, F) = nl.default_pair();
                vec![
                    (
                        match &input {
                            VarTensor::Advice { inner: advices, .. } => {
                                qlookup.clone() * cs.query_advice(advices[x], Rotation(0))
                                    + not_qlookup.clone() * default_x
                            }
                            VarTensor::Fixed { inner: fixed, .. } => {
                                qlookup.clone() * cs.query_fixed(fixed[x], Rotation(0))
                                    + not_qlookup.clone() * default_x
                            }
                            _ => panic!("uninitialized Vartensor"),
                        },
                        table.clone().borrow().table_input,
                    ),
                    (
                        match &output {
                            VarTensor::Advice { inner: advices, .. } => {
                                qlookup * cs.query_advice(advices[x], Rotation(0))
                                    + not_qlookup * default_y
                            }
                            VarTensor::Fixed { inner: fixed, .. } => {
                                qlookup * cs.query_fixed(fixed[x], Rotation(0))
                                    + not_qlookup * default_y
                            }
                            _ => panic!("uninitialized Vartensor"),
                        },
                        table.clone().borrow().table_output,
                    ),
                ]
            });
        }
        self.lookup_selectors.extend(selectors);
        // if we haven't previously initialized the input/output, do so now
        if let VarTensor::None = self.lookup_input {
            warn!("assiging lookup input");
            self.lookup_input = input.clone();
        }
        if let VarTensor::None = self.lookup_output {
            self.lookup_output = output.clone();
        }
        Ok(())
    }

    /// layout_tables must be called before layout.
    pub fn layout_tables(&mut self, layouter: &mut impl Layouter<F>) -> Result<(), Box<dyn Error>> {
        for (_, table) in &self.tables {
            if !table.borrow().is_assigned {
                table.borrow_mut().layout(layouter)?;
            }
        }
        Ok(())
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    /// * `offset` - Offset to assign.
    /// * `op` - The operation being represented.
    pub fn layout(
        &mut self,
        region: &mut Region<F>,
        values: &[ValTensor<F>],
        offset: &mut usize,
        op: OpKind,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let mut cp_values = vec![];
        for v in values.iter() {
            if let ValTensor::Instance { .. } = v {
                cp_values.push(layouts::identity(self, region, &[v.clone()], offset)?);
            } else {
                cp_values.push(v.clone());
            }
        }
        trace!("laying out {:?}", op);
        let res = match op {
            OpKind::Poly(op) => Some(match op {
                Op::Dot => layouts::dot(self, region, cp_values[..].try_into()?, offset)?,
                Op::Sum => layouts::sum(self, region, cp_values[..].try_into()?, offset)?,
                Op::Matmul => layouts::matmul(self, region, cp_values[..].try_into()?, offset)?,
                Op::Affine => layouts::affine(self, region, cp_values[..].try_into()?, offset)?,
                Op::Conv { padding, stride } => layouts::conv(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    padding,
                    stride,
                    offset,
                )?,
                Op::SumPool {
                    padding,
                    stride,
                    kernel_shape,
                } => layouts::sumpool(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    padding,
                    stride,
                    kernel_shape,
                    offset,
                )?,
                Op::Add => {
                    layouts::pairwise(self, region, cp_values[..].try_into()?, offset, BaseOp::Add)?
                }
                Op::Sub => {
                    layouts::pairwise(self, region, cp_values[..].try_into()?, offset, BaseOp::Sub)?
                }
                Op::Mult => layouts::pairwise(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    offset,
                    BaseOp::Mult,
                )?,
                Op::Identity => layouts::identity(self, region, cp_values[..].try_into()?, offset)?,
                Op::Reshape(d) | Op::Flatten(d) => layouts::reshape(cp_values[..].try_into()?, &d)?,
                Op::BatchNorm => {
                    layouts::scale_and_shift(self, region, cp_values[..].try_into()?, offset)?
                }
                Op::ScaleAndShift => {
                    layouts::scale_and_shift(self, region, cp_values[..].try_into()?, offset)?
                }
                Op::Pad(p1, p2) => {
                    if values.len() != 1 {
                        return Err(Box::new(TensorError::DimError));
                    }
                    let mut input = cp_values[0].clone();
                    input.pad((p1, p2))?;
                    input
                }
                Op::Pow(exp) => layouts::pow(self, region, cp_values[..].try_into()?, exp, offset)?,
                Op::Pack(base, scale) => {
                    layouts::pack(self, region, cp_values[..].try_into()?, base, scale, offset)?
                }
                Op::Rescaled { inner, scale } => {
                    if scale.len() != values.len() {
                        return Err(Box::new(TensorError::DimMismatch(
                            "rescaled inputs".to_string(),
                        )));
                    }

                    let res =
                        &layouts::rescale(self, region, cp_values[..].try_into()?, &scale, offset)?
                            [..];
                    self.layout(region, res, offset, OpKind::Poly(*inner))?
                        .unwrap()
                }
                Op::RangeCheck(tol) => {
                    layouts::range_check(self, region, cp_values[..].try_into()?, offset, tol)?
                }
                Op::GlobalSumPool => unreachable!(),
            }),
            OpKind::Lookup(nl) => match nl {
                LookupOp::PReLU { scale, .. } => Some(layouts::prelu(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    scale,
                    offset,
                )?),
                LookupOp::Mean { scale, .. } => Some(layouts::mean(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    scale,
                    offset,
                )?),
                _ => Some(layouts::nonlinearity(
                    self,
                    region,
                    cp_values[..].try_into()?,
                    nl,
                    offset,
                )?),
            },
            OpKind::Const => None,
            OpKind::Input => None,
            _ => {
                return Err(Box::new(CircuitError::UnsupportedOp));
            }
        };
        Ok(res)
    }
}
