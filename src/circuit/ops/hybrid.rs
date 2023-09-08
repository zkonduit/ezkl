use super::*;
use crate::{
    circuit::{self, layouts, utils, Tolerance},
    fieldutils::{felt_to_i128, i128_to_felt},
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
// import run args from model

#[allow(missing_docs)]
/// An enum representing the operations that consist of both lookups and arithmetic operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HybridOp {
    Abs,
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceArgMax {
        dim: usize,
    },
    MaxPool2d {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    ReduceArgMin {
        dim: usize,
    },
    Softmax {
        scale: utils::F32,
        axes: Vec<usize>,
    },
    RangeCheck(Tolerance),
    Greater,
    Less,
    Equals,
    Gather {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    TopK {
        dim: usize,
        k: usize,
    },
    GatherElements {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for HybridOp {
    ///
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        match self {
            HybridOp::Greater | HybridOp::Less | HybridOp::Equals => vec![0, 1],
            _ => vec![],
        }
    }

    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = inputs[0].clone().map(|x| felt_to_i128(x));

        let (res, intermediate_lookups) = match &self {
            HybridOp::Abs => (tensor::ops::abs(&x)?, vec![]),
            HybridOp::ReduceMax { axes, .. } => {
                let res = tensor::ops::max_axes(&x, axes)?;
                let max_minus_one =
                    Tensor::from(vec![x.clone().into_iter().max().unwrap() - 1].into_iter());
                let unit = Tensor::from(vec![1].into_iter());
                // relu(x - max(x - 1)
                let inter_1 = (x.clone() - max_minus_one)?;
                // relu(1 - sum(relu(inter_1)))
                let inter_2 = (unit
                    - tensor::ops::sum(&tensor::ops::nonlinearities::leakyrelu(&inter_1, 0.0))?)?;

                (res.clone(), vec![inter_1, inter_2])
            }
            HybridOp::ReduceMin { axes, .. } => {
                let res = tensor::ops::min_axes(&x, axes)?;
                let min_plus_one =
                    Tensor::from(vec![x.clone().into_iter().min().unwrap() + 1].into_iter());
                let unit = Tensor::from(vec![1].into_iter());
                // relu(min(x + 1) - x)
                let inter_1 = (min_plus_one - x.clone())?;
                // relu(1 - sum(relu(inter_1)))
                let inter_2 = (unit
                    - tensor::ops::sum(&tensor::ops::nonlinearities::leakyrelu(&inter_1, 0.0))?)?;
                (res.clone(), vec![inter_1, inter_2])
            }
            HybridOp::ReduceArgMax { dim } => {
                let res = tensor::ops::argmax_axes(&x, *dim)?;
                let mut inter_equals = vec![Tensor::from(0..x.dims()[*dim] as i128)];
                let inter =
                    Op::f(&HybridOp::ReduceMax { axes: vec![*dim] }, inputs)?.intermediate_lookups;
                inter_equals.extend(inter);

                (res.clone(), inter_equals)
            }
            HybridOp::ReduceArgMin { dim } => {
                let res = tensor::ops::argmin_axes(&x, *dim)?;
                let mut inter_equals = vec![Tensor::from(0..x.dims()[*dim] as i128)];
                let inter =
                    Op::f(&HybridOp::ReduceMin { axes: vec![*dim] }, inputs)?.intermediate_lookups;
                inter_equals.extend(inter);

                (res.clone(), inter_equals)
            }
            HybridOp::Gather { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    let res = tensor::ops::gather(&x, idx, *dim)?;
                    (res.clone(), vec![])
                } else {
                    let y = inputs[1].clone().map(|x| felt_to_i128(x));
                    let inter_equals: Vec<Tensor<i128>> =
                        vec![Tensor::from(0..x.dims()[*dim] as i128)];
                    let res = tensor::ops::gather(&x, &y.map(|x| x as usize), *dim)?;
                    (res.clone(), inter_equals)
                }
            }
            HybridOp::TopK { dim, k } => {
                let res = tensor::ops::topk_axes(&x, *k, *dim)?;

                let mut inter_equals = x
                    .clone()
                    .into_iter()
                    .flat_map(|elem| {
                        tensor::ops::equals(&res, &vec![elem].into_iter().into())
                            .unwrap()
                            .1
                    })
                    .collect::<Vec<_>>();

                // sort in descending order and take pairwise differences
                inter_equals.push(
                    x.into_iter()
                        .sorted()
                        .tuple_windows()
                        .map(|(a, b)| b - a)
                        .into(),
                );

                (res.clone(), inter_equals)
            }
            HybridOp::GatherElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    let res = tensor::ops::gather_elements(&x, idx, *dim)?;
                    (res.clone(), vec![])
                } else {
                    let y = inputs[1].clone().map(|x| felt_to_i128(x));
                    let inter_equals: Vec<Tensor<i128>> =
                        vec![Tensor::from(0..x.dims()[*dim] as i128)];
                    let res = tensor::ops::gather_elements(&x, &y.map(|x| x as usize), *dim)?;
                    (res.clone(), inter_equals)
                }
            }
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => (
                tensor::ops::max_pool2d(&x, padding, stride, pool_dims)?,
                vec![],
            ),
            HybridOp::Softmax { scale, axes } => {
                tensor::ops::nonlinearities::softmax_axes(&x, scale.into(), axes)
            }
            HybridOp::RangeCheck(tol) => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                (
                    tensor::ops::nonlinearities::range_check_percent(&[x, y], 128, 128, tol.val),
                    vec![],
                )
            }
            HybridOp::Greater => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::greater(&x, &y)?
            }
            HybridOp::Less => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::less(&x, &y)?
            }
            HybridOp::Equals => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::equals(&x, &y)?
            }
        };

        // convert back to felt
        let output = res.map(|x| i128_to_felt(x));

        Ok(ForwardResult {
            output,
            intermediate_lookups,
        })
    }

    fn as_string(&self) -> String {
        let name = match self {
            HybridOp::Abs => "ABS",
            HybridOp::ReduceMax { .. } => "REDUCEMAX",
            HybridOp::ReduceArgMax { .. } => "REDUCEARGMAX",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::ReduceMin { .. } => "REDUCEMIN",
            HybridOp::ReduceArgMin { .. } => "REDUCEARGMIN",
            HybridOp::Softmax { .. } => "SOFTMAX",
            HybridOp::RangeCheck(..) => "RANGECHECK",
            HybridOp::Greater { .. } => "GREATER",
            HybridOp::Less { .. } => "LESS",
            HybridOp::Equals => "EQUALS",
            HybridOp::Gather { .. } => "GATHER",
            HybridOp::TopK { .. } => "TOPK",
            HybridOp::GatherElements { .. } => "GATHERELEMENTS",
        };
        name.into()
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
        Ok(Some(match self {
            HybridOp::Abs => layouts::abs(config, region, values[..].try_into()?)?,
            HybridOp::Gather { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::gather(&values[0].get_inner_tensor()?, idx, *dim)?.into()
                } else {
                    layouts::gather(config, region, values[..].try_into()?, *dim)?
                }
            }
            HybridOp::GatherElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::gather_elements(&values[0].get_inner_tensor()?, idx, *dim)?.into()
                } else {
                    layouts::gather_elements(config, region, values[..].try_into()?, *dim)?
                }
            }
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => layouts::max_pool2d(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *stride,
                *pool_dims,
            )?,
            HybridOp::ReduceMax { axes } => {
                layouts::max_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::ReduceArgMax { dim } => {
                layouts::argmax_axes(config, region, values[..].try_into()?, *dim)?
            }
            HybridOp::ReduceMin { axes } => {
                layouts::min_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::ReduceArgMin { dim } => {
                layouts::argmin_axes(config, region, values[..].try_into()?, *dim)?
            }
            HybridOp::Softmax { scale, axes } => {
                layouts::softmax_axes(config, region, values[..].try_into()?, *scale, axes)?
            }
            HybridOp::RangeCheck(tol) => layouts::range_check_percent(
                config,
                region,
                values[..].try_into()?,
                tol.scale,
                tol.val,
            )?,
            HybridOp::Greater => layouts::greater(config, region, values[..].try_into()?)?,
            HybridOp::Less => layouts::less(config, region, values[..].try_into()?)?,
            HybridOp::Equals => layouts::equals(config, region, values[..].try_into()?)?,
            HybridOp::TopK { dim, k } => {
                layouts::topk_axes(config, region, values[..].try_into()?, *k, *dim)?
            }
        }))
    }

    fn requires_specific_input_scales(&self) -> Vec<(usize, u32)> {
        match self {
            HybridOp::Gather { .. } | HybridOp::GatherElements { .. } => vec![(1, 0)],
            _ => vec![],
        }
    }

    fn out_scale(&self, in_scales: Vec<u32>) -> u32 {
        match self {
            HybridOp::Greater { .. }
            | HybridOp::Less { .. }
            | HybridOp::ReduceArgMax { .. }
            | HybridOp::ReduceArgMin { .. } => 0,
            HybridOp::Softmax { .. } => 2 * in_scales[0],
            _ => in_scales[0],
        }
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            HybridOp::ReduceMax { .. }
            | HybridOp::ReduceMin { .. }
            | HybridOp::MaxPool2d { .. }
            | HybridOp::Abs => Op::<F>::required_lookups(&LookupOp::ReLU),
            HybridOp::Softmax { scale, .. } => {
                vec![
                    LookupOp::Exp { scale: *scale },
                    LookupOp::Recip {
                        scale: scale.0.powf(2.0).into(),
                    },
                ]
            }
            HybridOp::RangeCheck(tol) => {
                let mut lookups = vec![];
                if tol.val > 0.0 {
                    let scale_squared = tol.scale.0.powf(2.0);
                    lookups.extend([
                        LookupOp::Recip {
                            scale: scale_squared.into(),
                        },
                        LookupOp::GreaterThan {
                            a: circuit::utils::F32((tol.val * scale_squared) / 100.0),
                        },
                    ]);
                }
                lookups
            }
            HybridOp::Greater { .. }
            | HybridOp::Less { .. }
            | HybridOp::Equals
            | HybridOp::Gather { .. }
            | HybridOp::TopK { .. }
            | HybridOp::GatherElements { .. } => {
                vec![LookupOp::GreaterThan {
                    a: circuit::utils::F32(0.),
                }]
            }
            HybridOp::ReduceArgMax { .. } | HybridOp::ReduceArgMin { .. } => {
                vec![
                    LookupOp::ReLU,
                    LookupOp::GreaterThan {
                        a: circuit::utils::F32(0.),
                    },
                ]
            }
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
