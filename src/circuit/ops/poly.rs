use crate::{
    circuit::layouts,
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be expressed as arithmetic (non lookup) operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PolyOp<F: PrimeField + TensorType + PartialOrd> {
    Einsum {
        equation: String,
    },
    Conv {
        kernel: Tensor<F>,
        bias: Option<Tensor<F>>,
        padding: [(usize, usize); 2],
        stride: (usize, usize),
    },
    Downsample {
        axis: usize,
        stride: usize,
        modulo: usize,
    },
    DeConv {
        kernel: Tensor<F>,
        bias: Option<Tensor<F>>,
        padding: [(usize, usize); 2],
        output_padding: (usize, usize),
        stride: (usize, usize),
    },
    SumPool {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        kernel_shape: (usize, usize),
    },
    Add,
    Sub,
    Neg,
    Mult,
    Identity,
    Reshape(Vec<usize>),
    MoveAxis {
        source: usize,
        destination: usize,
    },
    Flatten(Vec<usize>),
    Pad([(usize, usize); 2]),
    Sum {
        axes: Vec<usize>,
    },
    Prod {
        axes: Vec<usize>,
        len_prod: usize,
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
    Not,
    And,
    Or,
    Xor,
}

impl<F: PrimeField + TensorType + PartialOrd> PolyOp<F> {}

impl<F: PrimeField + TensorType + PartialOrd + Serialize + for<'de> Deserialize<'de>> Op<F>
    for PolyOp<F>
{
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        match &self {
            PolyOp::MoveAxis { .. } => "MOVEAXIS".into(),
            PolyOp::Downsample { .. } => "DOWNSAMPLE".into(),
            PolyOp::Resize { .. } => "RESIZE".into(),
            PolyOp::Iff => "IFF".into(),
            PolyOp::Einsum { equation, .. } => format!("EINSUM {}", equation),
            PolyOp::Identity => "IDENTITY".into(),
            PolyOp::Reshape(_) => "RESHAPE".into(),
            PolyOp::Flatten(_) => "FLATTEN".into(),
            PolyOp::Pad(_) => "PAD".into(),
            PolyOp::Add => "ADD".into(),
            PolyOp::Mult => "MULT".into(),
            PolyOp::Sub => "SUB".into(),
            PolyOp::Sum { .. } => "SUM".into(),
            PolyOp::Prod { .. } => "PROD".into(),
            PolyOp::Pow(_) => "POW".into(),
            PolyOp::Pack(_, _) => "PACK".into(),
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL".into(),
            PolyOp::Conv { .. } => "CONV".into(),
            PolyOp::DeConv { .. } => "DECONV".into(),
            PolyOp::SumPool { .. } => "SUMPOOL".into(),
            PolyOp::Concat { .. } => "CONCAT".into(),
            PolyOp::Slice { .. } => "SLICE".into(),
            PolyOp::Neg => "NEG".into(),
            PolyOp::Not => "NOT".into(),
            PolyOp::And => "AND".into(),
            PolyOp::Or => "OR".into(),
            PolyOp::Xor => "XOR".into(),
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let mut inputs = inputs.to_vec();
        let res = match &self {
            PolyOp::And => tensor::ops::and(&inputs[0], &inputs[1]),
            PolyOp::Or => tensor::ops::or(&inputs[0], &inputs[1]),
            PolyOp::Xor => tensor::ops::xor(&inputs[0], &inputs[1]),
            PolyOp::Not => tensor::ops::not(&inputs[0]),
            PolyOp::Downsample {
                axis,
                stride,
                modulo,
            } => tensor::ops::downsample(&inputs[0], *axis, *stride, *modulo),
            PolyOp::Resize { scale_factor } => tensor::ops::resize(&inputs[0], scale_factor),
            PolyOp::Iff => tensor::ops::iff(&inputs[0], &inputs[1], &inputs[2]),
            PolyOp::Einsum { equation } => tensor::ops::einsum(equation, &inputs),
            PolyOp::Identity => Ok(inputs[0].clone()),
            PolyOp::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            PolyOp::MoveAxis {
                source,
                destination,
            } => inputs[0].move_axis(*source, *destination),
            PolyOp::Flatten(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims);
                Ok(t)
            }
            PolyOp::Pad(p) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pad inputs".to_string()));
                }
                tensor::ops::pad(&inputs[0], *p)
            }
            PolyOp::Add => tensor::ops::add(&inputs),
            PolyOp::Neg => tensor::ops::neg(&inputs[0]),
            PolyOp::Sub => tensor::ops::sub(&inputs),
            PolyOp::Mult => tensor::ops::mult(&inputs),
            PolyOp::Conv {
                kernel: a,
                bias,
                padding,
                stride,
            } => {
                inputs.push(a.clone());
                if let Some(b) = bias {
                    inputs.push(b.clone());
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
                inputs.push(a.clone());
                if let Some(b) = bias {
                    inputs.push(b.clone());
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

                tensor::ops::pack(&inputs[0], F::from(*base as u64), *scale)
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
            PolyOp::Prod { axes, .. } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("prod inputs".to_string()));
                }
                tensor::ops::prod_axes(&inputs[0], axes)
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => tensor::ops::concat(&inputs, *axis),
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
            PolyOp::Xor => layouts::xor(config, region, values[..].try_into()?)?,
            PolyOp::Or => layouts::or(config, region, values[..].try_into()?)?,
            PolyOp::And => layouts::and(config, region, values[..].try_into()?)?,
            PolyOp::Not => layouts::not(config, region, values[..].try_into()?)?,
            PolyOp::MoveAxis {
                source,
                destination,
            } => layouts::move_axis(values[..].try_into()?, *source, *destination)?,
            PolyOp::Downsample {
                axis,
                stride,
                modulo,
            } => layouts::downsample(config, region, values[..].try_into()?, axis, stride, modulo)?,
            PolyOp::Resize { scale_factor } => {
                layouts::resize(config, region, values[..].try_into()?, scale_factor)?
            }
            PolyOp::Neg => layouts::neg(config, region, values[..].try_into()?)?,
            PolyOp::Iff => layouts::iff(config, region, values[..].try_into()?)?,
            PolyOp::Einsum { equation } => layouts::einsum(config, region, &mut values, equation)?,
            PolyOp::Sum { axes } => {
                layouts::sum_axes(config, region, values[..].try_into()?, axes)?
            }
            PolyOp::Prod { axes, .. } => {
                layouts::prod_axes(config, region, values[..].try_into()?, axes)?
            }
            PolyOp::Conv {
                kernel,
                bias,
                padding,
                stride,
            } => {
                values.push(kernel.clone().into());
                if let Some(bias) = bias {
                    values.push(bias.clone().into());
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
                values.push(kernel.clone().into());
                if let Some(bias) = bias {
                    values.push(bias.clone().into());
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
            PolyOp::Add => layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Add)?,
            PolyOp::Sub => layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Sub)?,
            PolyOp::Mult => {
                layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Mult)?
            }
            PolyOp::Identity => layouts::identity(config, region, values[..].try_into()?)?,
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => layouts::reshape(values[..].try_into()?, d)?,
            PolyOp::Pad(p) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError));
                }
                let mut input = values[0].clone();
                input.pad(*p)?;
                input
            }
            PolyOp::Pow(exp) => layouts::pow(config, region, values[..].try_into()?, *exp)?,
            PolyOp::Pack(base, scale) => {
                layouts::pack(config, region, values[..].try_into()?, *base, *scale)?
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => layouts::concat(values[..].try_into()?, axis)?,
            PolyOp::Slice { axis, start, end } => {
                layouts::slice(config, region, values[..].try_into()?, axis, start, end)?
            }
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>) -> u32 {
        match self {
            PolyOp::Xor | PolyOp::Or | PolyOp::And | PolyOp::Not => 0,
            PolyOp::Neg => in_scales[0],
            PolyOp::MoveAxis { .. } => in_scales[0],
            PolyOp::Downsample { .. } => in_scales[0],
            PolyOp::Resize { .. } => in_scales[0],
            PolyOp::Iff => in_scales[1],
            PolyOp::Einsum { .. } => {
                let mut scale = in_scales[0];
                for s in in_scales.iter().skip(1) {
                    scale += *s;
                }
                scale
            }
            PolyOp::Prod { len_prod, .. } => in_scales[0] * (*len_prod as u32),
            PolyOp::Sum { .. } => in_scales[0],
            PolyOp::Conv { kernel, bias, .. } => {
                let kernel_scale = match kernel.scale() {
                    Some(s) => s,
                    None => panic!("scale must be set for conv kernel"),
                };
                let output_scale = in_scales[0] + kernel_scale;
                if let Some(b) = bias {
                    let bias_scale = match b.scale() {
                        Some(s) => s,
                        None => panic!("scale must be set for conv bias"),
                    };
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::DeConv { kernel, bias, .. } => {
                let kernel_scale = match kernel.scale() {
                    Some(s) => s,
                    None => panic!("scale must be set for deconv kernel"),
                };
                let output_scale = in_scales[0] + kernel_scale;
                if let Some(b) = bias {
                    let bias_scale = match b.scale() {
                        Some(s) => s,
                        None => panic!("scale must be set for deconv bias"),
                    };
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::SumPool { .. } => in_scales[0],
            PolyOp::Add => {
                let mut scale_a = 0;
                let scale_b = in_scales[0];
                scale_a += in_scales[1];
                assert_eq!(scale_a, scale_b);
                scale_a
            }
            PolyOp::Sub => in_scales[0],
            PolyOp::Mult => {
                let mut scale = in_scales[0];
                scale += in_scales[1];
                scale
            }
            PolyOp::Identity => in_scales[0],
            PolyOp::Reshape(_) | PolyOp::Flatten(_) => in_scales[0],
            PolyOp::Pad(_) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
            PolyOp::Concat { axis: _ } => in_scales[0],
            PolyOp::Slice { .. } => in_scales[0],
        }
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        if matches!(self, PolyOp::Add { .. } | PolyOp::Sub) {
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
