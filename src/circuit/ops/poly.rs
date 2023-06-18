use crate::{
    circuit::layouts,
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be expressed as arithmetic (non lookup) operations.
#[derive(Clone, Debug)]
pub enum PolyOp<F: PrimeField + TensorType + PartialOrd> {
    Einsum {
        equation: String,
    },
    Conv {
        kernel: ValTensor<F>,
        bias: Option<ValTensor<F>>,
        padding: (usize, usize),
        stride: (usize, usize),
    },
    DeConv {
        kernel: ValTensor<F>,
        bias: Option<ValTensor<F>>,
        padding: (usize, usize),
        output_padding: (usize, usize),
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
    Gather {
        dim: usize,
        index: Tensor<usize>,
    },
    Flatten(Vec<usize>),
    Pad(usize, usize),
    Sum {
        axes: Vec<usize>,
    },
    Pow(u32),
    Pack(u32, u32),
    GlobalSumPool,
    Concat {
        axis: usize,
    },
    Slice {
        axis: usize,
        start: usize,
        end: usize,
    },
    Iff,
    Resize {
        scale_factor: Vec<usize>,
    },
}

impl<F: PrimeField + TensorType + PartialOrd> PolyOp<F> {}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for PolyOp<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_string(&self) -> String {
        let name = match &self {
            PolyOp::Resize { .. } => "RESIZE",
            PolyOp::Iff => "IFF",
            PolyOp::Einsum { .. } => "EINSUM",
            PolyOp::Identity => "IDENTITY",
            PolyOp::Reshape(_) => "RESHAPE",
            PolyOp::Flatten(_) => "FLATTEN",
            PolyOp::Pad(_, _) => "PAD",
            PolyOp::Add { .. } => "ADD",
            PolyOp::Mult { .. } => "MULT",
            PolyOp::Sub => "SUB",
            PolyOp::Sum { .. } => "SUM",
            PolyOp::Pow(_) => "POW",
            PolyOp::Pack(_, _) => "PACK",
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL",
            PolyOp::Conv { .. } => "CONV",
            PolyOp::DeConv { .. } => "DECONV",
            PolyOp::SumPool { .. } => "SUMPOOL",
            PolyOp::Gather { .. } => "GATHER",
            PolyOp::Concat { .. } => "CONCAT",
            PolyOp::Slice { .. } => "SLICE",
        };
        name.into()
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<ForwardResult, TensorError> {
        let mut inputs = inputs.to_vec();
        let res = match &self {
            PolyOp::Resize { scale_factor } => tensor::ops::resize(&inputs[0], scale_factor),
            PolyOp::Iff => tensor::ops::iff(&inputs[0], &inputs[1], &inputs[2]),
            PolyOp::Einsum { equation } => tensor::ops::einsum(equation, &inputs),
            PolyOp::Gather { dim, index } => tensor::ops::gather(&inputs[0], *dim, index),
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
            PolyOp::Add { a } => {
                if let Some(a) = a {
                    inputs.push(Tensor::new(Some(&a.get_int_evals().unwrap()), a.dims())?);
                }
                tensor::ops::add(&inputs)
            }
            PolyOp::Sub => tensor::ops::sub(&inputs),
            PolyOp::Mult { a } => {
                if let Some(a) = a {
                    inputs.push(Tensor::new(Some(&a.get_int_evals().unwrap()), a.dims())?);
                }
                tensor::ops::mult(&inputs)
            }
            PolyOp::Conv {
                kernel: a,
                bias,
                padding,
                stride,
            } => {
                inputs.push(Tensor::new(Some(&a.get_int_evals().unwrap()), a.dims())?);
                if let Some(b) = bias {
                    inputs.push(Tensor::new(Some(&b.get_int_evals().unwrap()), b.dims())?);
                }
                tensor::ops::conv(&inputs, *padding, *stride)
            }
            PolyOp::DeConv {
                kernel: a,
                bias,
                padding,
                output_padding,
                stride,
            } => {
                inputs.push(Tensor::new(Some(&a.get_int_evals().unwrap()), a.dims())?);
                if let Some(b) = bias {
                    inputs.push(Tensor::new(Some(&b.get_int_evals().unwrap()), b.dims())?);
                }
                tensor::ops::deconv(&inputs, *padding, *output_padding, *stride)
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
            PolyOp::Sum { axes } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("sum inputs".to_string()));
                }
                tensor::ops::sum_axes(&inputs[0], axes)
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => {
                if inputs.len() < 2 {
                    return Err(TensorError::DimMismatch("concat inputs".to_string()));
                }
                tensor::ops::concat(&inputs, *axis)
            }
            PolyOp::Slice { axis, start, end } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("slice inputs".to_string()));
                }
                Ok(tensor::ops::slice(&inputs[0], axis, start, end)?)
            }
        }?;

        Ok(ForwardResult {
            output: res,
            intermediate_lookups: vec![],
        })
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let mut values = values.to_vec();

        Ok(Some(match self {
            PolyOp::Resize { scale_factor } => {
                layouts::resize(config, region, values[..].try_into()?, scale_factor)?
            }
            PolyOp::Iff => layouts::iff(config, region, values[..].try_into()?)?,
            PolyOp::Einsum { equation } => {
                let out = layouts::einsum(config, region, &mut values, equation)?;
                out
            }
            PolyOp::Gather { dim, index } => {
                tensor::ops::gather(&values[0].get_inner_tensor()?, *dim, index)?.into()
            }
            PolyOp::Sum { axes } => {
                layouts::sum_axes(config, region, values[..].try_into()?, axes)?
            }
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
                layouts::conv(config, region, values[..].try_into()?, *padding, *stride)?
            }
            PolyOp::DeConv {
                kernel,
                bias,
                padding,
                output_padding,
                stride,
            } => {
                values.push(kernel.clone());
                if let Some(bias) = bias {
                    values.push(bias.clone());
                }
                layouts::deconv(
                    config,
                    region,
                    values[..].try_into()?,
                    *padding,
                    *output_padding,
                    *stride,
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
            )?,
            PolyOp::Add { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }

                layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Add)?
            }
            PolyOp::Sub => layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Sub)?,
            PolyOp::Mult { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }
                layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Mult)?
            }
            PolyOp::Identity => layouts::identity(config, region, values[..].try_into()?)?,
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => layouts::reshape(values[..].try_into()?, d)?,
            PolyOp::Pad(p1, p2) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError));
                }
                let mut input = values[0].clone();
                input.pad((*p1, *p2))?;
                input
            }
            PolyOp::Pow(exp) => layouts::pow(config, region, values[..].try_into()?, *exp)?,
            PolyOp::Pack(base, scale) => {
                layouts::pack(config, region, values[..].try_into()?, *base, *scale)?
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => {
                if values.len() < 2 {
                    return Err(Box::new(TensorError::DimError));
                }
                layouts::concat(values[..].try_into()?, axis)?
            }
            PolyOp::Slice { axis, start, end } => {
                layouts::slice(config, region, values[..].try_into()?, axis, start, end)?
            }
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>, _g: u32) -> u32 {
        match self {
            PolyOp::Resize { .. } => in_scales[0],
            PolyOp::Iff => in_scales[1],
            PolyOp::Einsum { .. } => {
                let mut scale = in_scales[0];
                for s in in_scales.iter().skip(1) {
                    scale += *s;
                }
                scale
            }
            PolyOp::Gather { .. } => in_scales[0],

            PolyOp::Sum { .. } => in_scales[0],
            PolyOp::Conv { kernel, bias, .. } => {
                let output_scale = in_scales[0] + kernel.scale();
                if let Some(b) = bias {
                    assert_eq!(output_scale, b.scale());
                }
                output_scale
            }
            PolyOp::DeConv { kernel, bias, .. } => {
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
            PolyOp::Pad(_, _) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
            PolyOp::Concat { axis: _ } => in_scales[0],
            PolyOp::Slice { .. } => in_scales[0],
        }
    }

    fn rescale(&self, input_scales: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        let inputs_to_scale = self.requires_homogenous_input_scales();
        // creates a rescaled op if the inputs are not homogenous
        homogenize_input_scales::<F>(self.clone(), input_scales, inputs_to_scale).unwrap()
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        if matches!(
            self,
            PolyOp::Add { .. } | PolyOp::Sub | PolyOp::Mult { .. } | PolyOp::Einsum { .. }
        ) {
            vec![0, 1]
        } else if matches!(self, PolyOp::Iff) {
            vec![1, 2]
        } else {
            vec![]
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
