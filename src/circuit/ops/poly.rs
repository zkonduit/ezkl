use crate::{
    circuit::layouts,
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug)]
pub enum PolyOp<F: FieldExt + TensorType> {
    Dot,
    Matmul {
        a: Option<ValTensor<F>>,
    },
    Affine,
    Conv {
        kernel: ValTensor<F>,
        bias: Option<ValTensor<F>>,
        padding: (usize, usize),
        stride: (usize, usize),
    },
    SumPool {
        padding: (usize, usize),
        stride: (usize, usize),
        kernel_shape: (usize, usize),
    },
    Add {
        a: Option<ValTensor<F>>,
    },
    Sub,
    Mult {
        a: Option<ValTensor<F>>,
    },
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
    RangeCheck(i32),
}

impl<F: FieldExt + TensorType> Op<F> for PolyOp<F> {
    fn as_str(&self) -> &'static str {
        match &self {
            PolyOp::Identity => "IDENTITY",
            PolyOp::Reshape(_) => "RESHAPE",
            PolyOp::Flatten(_) => "FLATTEN",
            PolyOp::Pad(_, _) => "PAD",
            PolyOp::Add { .. } => "ADD",
            PolyOp::Mult { .. } => "MULT",
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
            PolyOp::Matmul { .. } => "MATMUL",
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
            PolyOp::Add { .. } => tensor::ops::add(inputs),
            PolyOp::Sub => tensor::ops::sub(inputs),
            PolyOp::Mult { .. } => tensor::ops::mult(inputs),
            PolyOp::Affine => tensor::ops::affine(inputs),
            PolyOp::BatchNorm => tensor::ops::scale_and_shift(inputs),
            PolyOp::ScaleAndShift => tensor::ops::scale_and_shift(inputs),
            PolyOp::Matmul { .. } => tensor::ops::matmul(inputs),
            PolyOp::Dot => tensor::ops::dot(&inputs.iter().collect()),
            PolyOp::Conv {
                padding, stride, ..
            } => tensor::ops::convolution(inputs, *padding, *stride),
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
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::RangeCheck(..) => Ok(inputs[0].clone()),
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: Option<&mut Region<F>>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let mut values = values.to_vec();

        Ok(Some(match self {
            PolyOp::Dot => layouts::dot(config, region, values[..].try_into()?, offset)?,
            PolyOp::Sum => layouts::sum(config, region, values[..].try_into()?, offset)?,
            PolyOp::Matmul { a } => {
                if let Some(a) = a {
                    let b = values;
                    values = vec![a.clone()];
                    values.extend(b);
                }

                layouts::matmul(config, region, values[..].try_into()?, offset)?
            }
            PolyOp::Affine => layouts::affine(config, region, values[..].try_into()?, offset)?,
            PolyOp::Conv {
                kernel,
                bias,
                padding,
                stride,
            } => {
                values.push(kernel.clone());
                if let Some(bias) = bias {
                    values.push(bias.clone());
                }
                layouts::conv(
                    config,
                    region,
                    values[..].try_into()?,
                    *padding,
                    *stride,
                    offset,
                )?
            }
            PolyOp::SumPool {
                padding,
                stride,
                kernel_shape,
            } => layouts::sumpool(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *stride,
                *kernel_shape,
                offset,
            )?,
            PolyOp::Add { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }

                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Add)?
            }
            PolyOp::Sub => {
                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Sub)?
            }
            PolyOp::Mult { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }
                layouts::pairwise(config, region, values[..].try_into()?, offset, BaseOp::Mult)?
            }
            PolyOp::Identity => layouts::identity(config, region, values[..].try_into()?, offset)?,
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => layouts::reshape(values[..].try_into()?, d)?,
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
            PolyOp::RangeCheck(tol) => {
                layouts::range_check(config, region, values[..].try_into()?, offset, *tol)?
            }
            PolyOp::GlobalSumPool => unreachable!(),
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>, _g: u32) -> u32 {
        match self {
            PolyOp::Dot => in_scales[0] + in_scales[1],
            PolyOp::Sum => in_scales[0],
            PolyOp::Matmul { a } => {
                let mut scale = in_scales[0];
                if let Some(a) = a {
                    scale += a.scale();
                } else {
                    scale += in_scales[1];
                }
                scale
            }
            PolyOp::Affine => in_scales[0] + in_scales[1],
            PolyOp::Conv { kernel, bias, .. } => {
                let output_scale = in_scales[0] + kernel.scale();
                if let Some(b) = bias {
                    assert_eq!(output_scale, b.scale());
                }
                output_scale
            }
            PolyOp::SumPool { .. } => in_scales[0],
            PolyOp::Add { a } => {
                let mut scale_a = 0;
                let scale_b = in_scales[0];
                if let Some(a) = a {
                    scale_a += a.scale();
                } else {
                    scale_a += in_scales[1];
                }
                assert_eq!(scale_a, scale_b);
                scale_a
            }
            PolyOp::Sub => in_scales[0],
            PolyOp::Mult { a } => {
                let mut scale = in_scales[0];
                if let Some(a) = a {
                    scale += a.scale();
                } else {
                    scale += in_scales[1];
                }
                scale
            }
            PolyOp::Identity => in_scales[0],
            PolyOp::Reshape(_) | PolyOp::Flatten(_) => in_scales[0],
            PolyOp::BatchNorm => 2 * in_scales[0],
            PolyOp::ScaleAndShift => 2 * in_scales[0],
            PolyOp::Pad(_, _) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::RangeCheck(_) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
        }
    }

    fn has_3d_input(&self) -> bool {
        matches!(
            self,
            PolyOp::Conv { .. }
                | PolyOp::SumPool { .. }
                | PolyOp::GlobalSumPool
                | PolyOp::Pad { .. }
        )
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn requires_homogenous_input_scales(&self) -> bool {
        matches!(self, PolyOp::Add { .. } | PolyOp::Sub)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
