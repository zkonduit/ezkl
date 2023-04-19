use std::fmt;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    circuit::layouts,
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum PolyOp {
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
        inner: Box<PolyOp>,
        scale: Vec<(usize, usize)>,
    },
    RangeCheck(i32),
}

impl PolyOp {
    /// circuit shape
    pub fn circuit_shapes(&self, input_shapes: Vec<Vec<usize>>) -> Vec<usize> {
        let mut shapes = match &self {
            PolyOp::Identity => vec![0, input_shapes[0].iter().product()],
            PolyOp::Reshape(_) => vec![0; 2],
            PolyOp::Flatten(_) => vec![0; 2],
            PolyOp::Pad(_, _) => vec![0; 2],
            PolyOp::Add => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Mult => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Sub => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Sum => vec![0, input_shapes[0].iter().product()],
            PolyOp::Dot => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Pow(_) => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Pack(_, _) => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::GlobalSumPool => unreachable!("should be handled by sumpool"),
            PolyOp::ScaleAndShift => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::BatchNorm => input_shapes.iter().map(|x| x.iter().product()).collect(),
            PolyOp::Conv { padding, stride } => {
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
                let op = PolyOp::Matmul;
                let output_len = op.circuit_shapes(input_shapes);

                vec![*output_len.last().unwrap(); 2]
            }
            PolyOp::SumPool {
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
                let op = PolyOp::Matmul;
                let output_len = op.circuit_shapes(input_shapes);

                vec![*output_len.last().unwrap(); 2]
            }
            PolyOp::Affine => {
                let s = input_shapes;
                // add 1 cause of bias
                let output_len = s[1][0] * (s[1][1] + 1);
                vec![output_len; 2]
            }
            PolyOp::Matmul => {
                let output_len = input_shapes[0].iter().product::<usize>() * input_shapes[1][1];

                vec![output_len; 2]
            }
            PolyOp::Rescaled { inner, .. } => inner.circuit_shapes(input_shapes),
            PolyOp::RangeCheck(..) => input_shapes.iter().map(|x| x.iter().product()).collect(),
        };
        match shapes.last() {
            // add output
            Some(s) => shapes.push(*s),
            _ => {}
        };
        shapes
    }
}

impl<F: FieldExt + TensorType> Op<F> for PolyOp {
    fn as_str(&self) -> &'static str {
        match &self {
            PolyOp::Identity => "IDENTITY",
            PolyOp::Reshape(_) => "RESHAPE",
            PolyOp::Flatten(_) => "FLATTEN",
            PolyOp::Pad(_, _) => "PAD",
            PolyOp::Add => "ADD",
            PolyOp::Mult => "MULT",
            PolyOp::Sub => "SUB",
            PolyOp::Sum => "SUM",
            PolyOp::Dot => "DOT",
            PolyOp::Pow(_) => "POW",
            PolyOp::Pack(_, _) => "PACK",
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL",
            PolyOp::ScaleAndShift => "SCALESHIFT",
            PolyOp::BatchNorm => "BATCHNORM",
            PolyOp::Conv { .. } => "CONV",
            PolyOp::SumPool { .. } => "SUMPOOL",
            PolyOp::Affine => "AFFINE",
            PolyOp::Matmul => "MATMUL",
            PolyOp::Rescaled { inner, .. } => Op::<F>::as_str(&**inner),
            PolyOp::RangeCheck(..) => "RANGECHECK",
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            PolyOp::Identity => Ok(inputs[0].clone()),
            PolyOp::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            PolyOp::Flatten(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            PolyOp::Pad(dim1, dim2) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pad inputs".to_string()));
                }
                tensor::ops::pad(&inputs[0], (*dim1, *dim2))
            }
            PolyOp::Add => tensor::ops::add(&inputs),
            PolyOp::Sub => tensor::ops::sub(&inputs),
            PolyOp::Mult => tensor::ops::mult(&inputs),
            PolyOp::Affine => tensor::ops::affine(&inputs),
            PolyOp::BatchNorm => tensor::ops::scale_and_shift(&inputs),
            PolyOp::ScaleAndShift => tensor::ops::scale_and_shift(&inputs),
            PolyOp::Matmul => tensor::ops::matmul(&inputs),
            PolyOp::Dot => tensor::ops::dot(&inputs.iter().collect()),
            PolyOp::Conv { padding, stride } => {
                tensor::ops::convolution(&inputs, *padding, *stride)
            }
            PolyOp::SumPool {
                padding,
                stride,
                kernel_shape,
            } => tensor::ops::sumpool(&inputs[0], *padding, *stride, *kernel_shape),
            PolyOp::Pack(base, scale) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pack inputs".to_string()));
                }

                tensor::ops::pack(&inputs[0], *base as i128, *scale)
            }
            PolyOp::Pow(u) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pow inputs".to_string()));
                }
                inputs[0].pow(*u)
            }
            PolyOp::Sum => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("sum inputs".to_string()));
                }
                tensor::ops::sum(&inputs[0])
            }
            PolyOp::Rescaled { inner, scale } => {
                if scale.len() != inputs.len() {
                    return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
                }

                let mut rescaled_inputs = vec![];
                let inputs = &mut inputs.to_vec();
                for (i, ri) in inputs.iter_mut().enumerate() {
                    rescaled_inputs.push(tensor::ops::rescale(ri, scale[i].1)?);
                }
                Ok(Op::<F>::f(&**inner, &rescaled_inputs)?)
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::RangeCheck(..) => Ok(inputs[0].clone()),
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut Region<F>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(Some(match self {
            PolyOp::Dot => layouts::dot(config, region, values[..].try_into()?, offset)?,
            PolyOp::Sum => layouts::sum(config, region, values[..].try_into()?, offset)?,
            PolyOp::Matmul => layouts::matmul(config, region, values[..].try_into()?, offset)?,
            PolyOp::Affine => layouts::affine(config, region, values[..].try_into()?, offset)?,
            PolyOp::Conv { padding, stride } => layouts::conv(
                config,
                region,
                values[..].try_into()?,
                padding.clone(),
                stride.clone(),
                offset,
            )?,
            PolyOp::SumPool {
                padding,
                stride,
                kernel_shape,
            } => layouts::sumpool(
                config,
                region,
                values[..].try_into()?,
                padding.clone(),
                stride.clone(),
                kernel_shape.clone(),
                offset,
            )?,
            PolyOp::Add => {
                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Add)?
            }
            PolyOp::Sub => {
                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Sub)?
            }
            PolyOp::Mult => {
                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Mult)?
            }
            PolyOp::Identity => layouts::identity(config, region, values[..].try_into()?, offset)?,
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => {
                layouts::reshape(values[..].try_into()?, &d)?
            }
            PolyOp::BatchNorm => {
                layouts::scale_and_shift(config, region, values[..].try_into()?, offset)?
            }
            PolyOp::ScaleAndShift => {
                layouts::scale_and_shift(config, region, values[..].try_into()?, offset)?
            }
            PolyOp::Pad(p1, p2) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError));
                }
                let mut input = values[0].clone();
                input.pad((*p1, *p2))?;
                input
            }
            PolyOp::Pow(exp) => layouts::pow(config, region, values[..].try_into()?, *exp, offset)?,
            PolyOp::Pack(base, scale) => layouts::pack(
                config,
                region,
                values[..].try_into()?,
                *base,
                *scale,
                offset,
            )?,
            PolyOp::Rescaled { inner, scale } => {
                if scale.len() != values.len() {
                    return Err(Box::new(TensorError::DimMismatch(
                        "rescaled inputs".to_string(),
                    )));
                }

                let res =
                    &layouts::rescale(config, region, values[..].try_into()?, &scale, offset)?[..];
                inner.layout(config, region, res, offset)?.unwrap()
            }
            PolyOp::RangeCheck(tol) => {
                layouts::range_check(config, region, values[..].try_into()?, offset, *tol)?
            }
            PolyOp::GlobalSumPool => unreachable!(),
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>, global_scale: u32) -> u32 {
        match self {
            PolyOp::Dot => in_scales[0] + in_scales[1],
            PolyOp::Sum => in_scales[0],
            PolyOp::Matmul => in_scales[0] + in_scales[1],
            PolyOp::Affine => in_scales[0] + in_scales[1],
            PolyOp::Conv { .. } => in_scales[0] + in_scales[1],
            PolyOp::SumPool { .. } => in_scales[0],
            PolyOp::Add => in_scales[0],
            PolyOp::Sub => in_scales[0],
            PolyOp::Mult => in_scales[0] + in_scales[1],
            PolyOp::Identity => in_scales[0],
            PolyOp::Reshape(_) | PolyOp::Flatten(_) => in_scales[0],
            PolyOp::BatchNorm => 2 * in_scales[0],
            PolyOp::ScaleAndShift => 2 * in_scales[0],
            PolyOp::Pad(_, _) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::Rescaled { inner, .. } => Op::<F>::out_scale(&**inner, in_scales, global_scale),
            PolyOp::RangeCheck(_) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
        }
    }

    fn out_dims(&self, in_dims: Vec<Vec<usize>>) -> Vec<usize> {
        match self {
            PolyOp::Dot => vec![1],
            PolyOp::Sum => vec![1],
            PolyOp::Matmul => {
                let a_dims = in_dims[0].clone();
                let b_dims = in_dims[1].clone();
                let mut dims = Vec::from(&a_dims[0..a_dims.len() - 2]);
                dims.push(a_dims[a_dims.len() - 2]);
                dims.push(b_dims[a_dims.len() - 1]);
                dims
            }
            PolyOp::Affine => {
                let weight_node = &in_dims[1];
                let out_dim = weight_node.clone()[0];
                vec![out_dim]
            }
            PolyOp::Conv { padding, stride } => {
                let oihw = in_dims[1].clone();
                let (out_channels, _, kernel_height, kernel_width) =
                    (oihw[0], oihw[1], oihw[2], oihw[3]);

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding.0, padding.1, stride.0, stride.1);

                println!("in_dims: {:?}", in_dims);

                let input_height = in_dims[0][1];
                let input_width = in_dims[0][2];

                let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                vec![out_channels, out_height, out_width]
            }
            PolyOp::SumPool {
                padding,
                stride,
                kernel_shape,
            } => {
                let (input_channels, kernel_height, kernel_width) =
                    (in_dims[0][0], kernel_shape.0, kernel_shape.1);

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding.0, padding.1, stride.0, stride.1);

                let input_height = in_dims[0][1];
                let input_width = in_dims[0][2];

                let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                vec![input_channels, out_height, out_width]
            }
            PolyOp::Add => in_dims[0].clone(),
            PolyOp::Sub => in_dims[0].clone(),
            PolyOp::Mult => in_dims[0].clone(),
            PolyOp::Identity => in_dims[0].clone(),
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => d.clone(),
            PolyOp::BatchNorm => in_dims[0].clone(),
            PolyOp::ScaleAndShift => in_dims[0].clone(),
            PolyOp::Pad(padding_h, padding_w) => {
                let input_channels = in_dims[0][0];

                let out_height = in_dims[0][1] + 2 * padding_h;
                let out_width = in_dims[0][2] + 2 * padding_w;
                vec![input_channels, out_height, out_width]
            }
            PolyOp::Pow(_) => in_dims[0].clone(),
            PolyOp::Pack(_, _) => vec![1],
            PolyOp::Rescaled { inner, .. } => Op::<F>::out_dims(&**inner, in_dims),
            PolyOp::RangeCheck(_) => in_dims[0].clone(),
            PolyOp::GlobalSumPool => {
                let input_channels = in_dims[0][0];
                vec![input_channels, 1, 1]
            }
        }
    }

    fn has_3d_input(&self) -> bool {
        match self {
            PolyOp::Conv { .. } => true,
            PolyOp::SumPool { .. } => true,
            PolyOp::GlobalSumPool => true,
            PolyOp::Pad { .. } => true,
            _ => false,
        }
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn requires_homogenous_input_scales(&self) -> bool {
        match self {
            PolyOp::Add | PolyOp::Sub => true,
            _ => false,
        }
    }

    fn bias_variable(&self) -> Option<usize> {
        match self {
            PolyOp::Affine | PolyOp::ScaleAndShift | PolyOp::Conv { .. } => Some(2),
            _ => None,
        }
    }

    fn required_poly(&self) -> Option<PolyOp> {
        Some(self.clone())
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

impl fmt::Display for PolyOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolyOp::Identity => write!(f, "identity"),
            PolyOp::Reshape(new_dims) => write!(f, "reshape to {:?}", new_dims),
            PolyOp::Flatten(new_dims) => write!(f, "flatten to {:?}", new_dims),
            PolyOp::Pad(dim1, dim2) => write!(f, "padding: ({:?}, {:?})", dim1, dim2),
            PolyOp::Add => write!(f, "add"),
            PolyOp::Sub => write!(f, "sub"),
            PolyOp::Sum => write!(f, "sum"),
            PolyOp::Mult => write!(f, "mult"),
            PolyOp::Matmul => write!(f, "matmul"),
            PolyOp::Dot => write!(f, "dot"),
            PolyOp::Pack(base, _) => write!(f, "pack with base {:?}", base),
            PolyOp::Affine => write!(f, "affine"),
            PolyOp::BatchNorm => write!(f, "batchnorm"),
            PolyOp::ScaleAndShift => write!(f, "scale & shift"),
            PolyOp::Conv { padding, stride } => {
                write!(f, "conv w/ padding: {:?}, stride: {:?}", padding, stride)
            }
            PolyOp::SumPool {
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
            PolyOp::GlobalSumPool => write!(f, "globalsumpool"),
            PolyOp::Pow(s) => write!(f, "pow {}", s),
            PolyOp::Rescaled { inner, scale } => {
                write!(
                    f,
                    "rescaled {} w/ scalings: {:?}",
                    **inner,
                    scale.iter().map(|e| e.1).collect_vec()
                )
            }
            PolyOp::RangeCheck(tol) => write!(f, "range check w/ tol {}", tol),
        }
    }
}
