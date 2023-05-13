use itertools::Itertools;

use crate::{
    circuit::layouts,
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be expressed as arithmetic (non lookup) operations.
#[derive(Clone, Debug)]
pub enum PolyOp<F: PrimeField + TensorType + PartialOrd> {
    Dot,
    Matmul {
        a: Option<ValTensor<F>>,
    },
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
    Iff
}

impl<F: PrimeField + TensorType + PartialOrd> PolyOp<F> {
    fn homogenize_input_scales(
        &self,
        input_scales: Vec<u32>,
        inputs_to_scale: Vec<usize>,
    ) -> Result<Box<dyn Op<F>>, Box<dyn Error>> {
        if inputs_to_scale.is_empty() {
            return Ok(Box::new(self.clone()));
        }

        let mut multipliers: Vec<u128> = vec![1; input_scales.len()];
        if !input_scales.windows(2).all(|w| w[0] == w[1]) {
            let max_scale = input_scales.iter().max().unwrap();
            let _ = input_scales
                .iter()
                .enumerate()
                .map(|(idx, input_scale)| {
                    if !inputs_to_scale.contains(&idx) {
                        return;
                    }
                    let scale_diff = max_scale - input_scale;
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        multipliers[idx] = mult as u128;
                    }
                })
                .collect_vec();
        }

        // only rescale if need to
        if multipliers.iter().any(|&x| x > 1) {
            Ok(Box::new(crate::circuit::Rescaled {
                inner: Box::new(self.clone()),
                scale: (0..input_scales.len()).zip(multipliers).collect_vec(),
            }))
        } else {
            Ok(Box::new(self.clone()))
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for PolyOp<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_str(&self) -> &'static str {
        match &self {
            PolyOp::Identity => "IDENTITY",
            PolyOp::Reshape(_) => "RESHAPE",
            PolyOp::Flatten(_) => "FLATTEN",
            PolyOp::Pad(_, _) => "PAD",
            PolyOp::Add { .. } => "ADD",
            PolyOp::Mult { .. } => "MULT",
            PolyOp::Sub => "SUB",
            PolyOp::Sum { .. } => "SUM",
            PolyOp::Dot => "DOT",
            PolyOp::Pow(_) => "POW",
            PolyOp::Pack(_, _) => "PACK",
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL",
            PolyOp::Conv { .. } => "CONV",
            PolyOp::SumPool { .. } => "SUMPOOL",
            PolyOp::Matmul { .. } => "MATMUL",
            PolyOp::Iff => "IFF",
            PolyOp::Gather { .. } => "GATHER",
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        let mut inputs = inputs.to_vec();
        match &self {
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
            PolyOp::Matmul { a } => {
                if let Some(a) = a {
                    let b = inputs;
                    inputs = vec![Tensor::new(Some(&a.get_int_evals().unwrap()), a.dims())?];
                    inputs.extend(b);
                }

                tensor::ops::matmul(&inputs)
            }
            PolyOp::Dot => tensor::ops::dot(&inputs.iter().collect()),
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
            PolyOp::Iff => {
                let mask = inputs[0].clone();
                // if mask > 0 then output a else output b
                let a = inputs[2].clone();
                let b = inputs[1].clone();

                let out = (mask.clone() * a.clone())?
                    - ((Tensor::from(vec![1_i128].into_iter()) - mask.clone())? * b.clone())?;

                Ok(out?)
            }
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut Option<&mut Region<F>>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let mut values = values.to_vec();

        Ok(Some(match self {
            PolyOp::Gather { dim, index } => {
                tensor::ops::gather(&values[0].get_inner_tensor()?, *dim, index)?.into()
            }
            PolyOp::Iff => layouts::iff(config, region, values[..].try_into()?, offset)?,
            PolyOp::Dot => layouts::dot(config, region, values[..].try_into()?, offset)?,
            PolyOp::Sum { axes } => {
                layouts::sum_axes(config, region, values[..].try_into()?, axes, offset)?
            }
            PolyOp::Matmul { a } => {
                if let Some(a) = a {
                    let b = values;
                    values = vec![a.clone()];
                    values.extend(b);
                }

                layouts::matmul(config, region, values[..].try_into()?, offset)?
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
            PolyOp::GlobalSumPool => unreachable!(),
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>, _g: u32) -> u32 {
        match self {
            PolyOp::Gather { .. } => in_scales[0],
            PolyOp::Iff => in_scales[1],
            PolyOp::Dot => in_scales[0] + in_scales[1],
            PolyOp::Sum { .. } => in_scales[0],
            PolyOp::Matmul { a } => {
                let mut scale = in_scales[0];
                if let Some(a) = a {
                    scale += a.scale();
                } else {
                    scale += in_scales[1];
                }
                scale
            }
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
            PolyOp::Pad(_, _) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
        }
    }

    fn rescale(&self, input_scales: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        let inputs_to_scale = self.requires_homogenous_input_scales();
        // creates a rescaled op if the inputs are not homogenous
        self.homogenize_input_scales(input_scales.clone(), inputs_to_scale)
            .unwrap()
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
