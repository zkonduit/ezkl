use super::scale_to_multiplier;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use super::utilities::node_output_shapes;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use super::VarScales;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use super::Visibility;

use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::circuit::CircuitError;
use crate::circuit::Constant;
use crate::circuit::Input;
use crate::circuit::Op;
use crate::circuit::Unknown;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use crate::graph::errors::GraphError;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use crate::graph::new_op_from_onnx;

use crate::tensor::TensorError;

use halo2curves::bn256::Fr as Fp;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use log::trace;

use serde::Deserialize;
use serde::Serialize;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use std::collections::BTreeMap;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use std::fmt;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tabled::Tabled;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tract_onnx::{
    self,
    prelude::{Node as OnnxNode, SymbolValues, TypedFact, TypedOp},
};

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{:?}", v)
    } else {
        String::new()
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
fn display_opkind(v: &SupportedOp) -> String {
    v.as_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rescaled {
    pub inner: Box<SupportedOp>,
    pub scale: Vec<(usize, u128)>,
}

impl Op<Fp> for Rescaled {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_string(&self) -> String {
        format!("RESCALED INPUT ({})", self.inner.as_string())
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        let in_scales = in_scales
            .into_iter()
            .zip(self.scale.iter())
            .map(|(a, b)| a + crate::graph::multiplier_to_scale(b.1 as f64))
            .collect();

        Op::<Fp>::out_scale(&*self.inner, in_scales)
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, CircuitError> {
        if self.scale.len() != values.len() {
            return Err(TensorError::DimMismatch("rescaled inputs".to_string()).into());
        }

        let res =
            &crate::circuit::layouts::rescale(config, region, values[..].try_into()?, &self.scale)?
                [..];
        self.inner.layout(config, region, res)
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebaseScale {
    pub inner: Box<SupportedOp>,
    pub rebase_op: HybridOp,
    pub target_scale: i32,
    pub original_scale: i32,
    pub multiplier: f64,
}

impl RebaseScale {
    pub fn rebase(
        inner: SupportedOp,
        global_scale: crate::Scale,
        op_out_scale: crate::Scale,
        scale_rebase_multiplier: u32,
    ) -> SupportedOp {
        if (op_out_scale > (global_scale * scale_rebase_multiplier as i32))
            && !inner.is_constant()
            && !inner.is_input()
            && !inner.is_identity()
        {
            let multiplier =
                scale_to_multiplier(op_out_scale - global_scale * scale_rebase_multiplier as i32);
            if let Some(op) = inner.get_rebased() {
                let multiplier = op.multiplier * multiplier;
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier,
                    rebase_op: HybridOp::Div {
                        denom: crate::circuit::utils::F32((multiplier) as f32),
                    },
                    original_scale: op.original_scale,
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale: global_scale * scale_rebase_multiplier as i32,
                    multiplier,
                    rebase_op: HybridOp::Div {
                        denom: crate::circuit::utils::F32(multiplier as f32),
                    },
                    original_scale: op_out_scale,
                })
            }
        } else {
            inner
        }
    }

    pub fn rebase_up(
        inner: SupportedOp,
        target_scale: crate::Scale,
        op_out_scale: crate::Scale,
    ) -> SupportedOp {
        if (op_out_scale < (target_scale)) && !inner.is_constant() && !inner.is_input() {
            let multiplier = scale_to_multiplier(op_out_scale - target_scale);
            if let Some(op) = inner.get_rebased() {
                let multiplier = op.multiplier * multiplier;
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier,
                    original_scale: op.original_scale,
                    rebase_op: HybridOp::Div {
                        denom: crate::circuit::utils::F32((multiplier) as f32),
                    },
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale,
                    multiplier,
                    original_scale: op_out_scale,
                    rebase_op: HybridOp::Div {
                        denom: crate::circuit::utils::F32(multiplier as f32),
                    },
                })
            }
        } else {
            inner
        }
    }
}

impl Op<Fp> for RebaseScale {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_string(&self) -> String {
        format!(
            "REBASED (div={:?}, rebasing_op={}) ({})",
            self.multiplier,
            <HybridOp as Op<Fp>>::as_string(&self.rebase_op),
            self.inner.as_string()
        )
    }

    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(self.target_scale)
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, CircuitError> {
        let original_res = self
            .inner
            .layout(config, region, values)?
            .ok_or(CircuitError::MissingLayout(self.as_string()))?;
        self.rebase_op.layout(config, region, &[original_res])
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SupportedOp {
    Linear(PolyOp),
    Nonlinear(LookupOp),
    Hybrid(HybridOp),
    Input(Input),
    Constant(Constant<Fp>),
    Unknown(Unknown),
    Rescaled(Rescaled),
    RebaseScale(RebaseScale),
}

impl SupportedOp {
    pub fn is_lookup(&self) -> bool {
        match self {
            SupportedOp::Nonlinear(_) => true,
            SupportedOp::RebaseScale(op) => op.inner.is_lookup(),
            _ => false,
        }
    }

    pub fn get_input(&self) -> Option<Input> {
        match self {
            SupportedOp::Input(op) => Some(op.clone()),
            _ => None,
        }
    }

    pub fn get_rebased(&self) -> Option<&RebaseScale> {
        match self {
            SupportedOp::RebaseScale(op) => Some(op),
            _ => None,
        }
    }

    pub fn get_lookup(&self) -> Option<&LookupOp> {
        match self {
            SupportedOp::Nonlinear(op) => Some(op),
            _ => None,
        }
    }

    pub fn get_constant(&self) -> Option<&Constant<Fp>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    pub fn get_mutable_constant(&mut self) -> Option<&mut Constant<Fp>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    fn homogenous_rescale(
        &self,
        in_scales: Vec<crate::Scale>,
    ) -> Result<Box<dyn Op<Fp>>, GraphError> {
        let inputs_to_scale = self.requires_homogenous_input_scales();
        let op = self.clone_dyn();
        super::homogenize_input_scales(op, in_scales, inputs_to_scale)
    }

    fn as_op(&self) -> &dyn Op<Fp> {
        match self {
            SupportedOp::Linear(op) => op,
            SupportedOp::Nonlinear(op) => op,
            SupportedOp::Hybrid(op) => op,
            SupportedOp::Input(op) => op,
            SupportedOp::Constant(op) => op,
            SupportedOp::Unknown(op) => op,
            SupportedOp::Rescaled(op) => op,
            SupportedOp::RebaseScale(op) => op,
        }
    }

    pub fn is_identity(&self) -> bool {
        match self {
            SupportedOp::Linear(op) => matches!(op, PolyOp::Identity { .. }),
            SupportedOp::Rescaled(op) => op.inner.is_identity(),
            SupportedOp::RebaseScale(op) => op.inner.is_identity(),
            _ => false,
        }
    }
}

impl From<Box<dyn Op<Fp>>> for SupportedOp {
    fn from(value: Box<dyn Op<Fp>>) -> Self {
        if let Some(op) = value.as_any().downcast_ref::<PolyOp>() {
            return SupportedOp::Linear(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<LookupOp>() {
            return SupportedOp::Nonlinear(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<HybridOp>() {
            return SupportedOp::Hybrid(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Input>() {
            return SupportedOp::Input(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Constant<Fp>>() {
            return SupportedOp::Constant(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Unknown>() {
            return SupportedOp::Unknown(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Rescaled>() {
            return SupportedOp::Rescaled(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<RebaseScale>() {
            return SupportedOp::RebaseScale(op.clone());
        };

        log::error!("Unsupported op type");
        log::warn!("defaulting to Unknown");
        SupportedOp::Unknown(Unknown {})
    }
}

impl Op<Fp> for SupportedOp {
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, CircuitError> {
        self.as_op().layout(config, region, values)
    }

    fn is_input(&self) -> bool {
        self.as_op().is_input()
    }

    fn is_constant(&self) -> bool {
        self.as_op().is_constant()
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        self.as_op().requires_homogenous_input_scales()
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        self.as_op().clone_dyn()
    }

    fn as_string(&self) -> String {
        self.as_op().as_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        self.as_op().out_scale(in_scales)
    }
}

pub type Outlet = (usize, usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub opkind: SupportedOp,
    pub out_scale: i32,
    pub inputs: Vec<Outlet>,
    pub out_dims: Vec<usize>,
    pub idx: usize,
    pub num_uses: usize,
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl Tabled for Node {
    const LENGTH: usize = 6;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        let mut headers = Vec::with_capacity(Self::LENGTH);
        for i in ["idx", "opkind", "out_scale", "inputs", "out_dims"] {
            headers.push(std::borrow::Cow::Borrowed(i));
        }
        headers
    }

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        let mut fields = Vec::with_capacity(Self::LENGTH);
        fields.push(std::borrow::Cow::Owned(self.idx.to_string()));
        fields.push(std::borrow::Cow::Owned(display_opkind(&self.opkind)));
        fields.push(std::borrow::Cow::Owned(self.out_scale.to_string()));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.inputs)));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.out_dims)));
        fields
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        (self.out_scale == other.out_scale)
            && (self.inputs == other.inputs)
            && (self.out_dims == other.out_dims)
            && (self.idx == other.idx)
            && (self.opkind.as_string() == other.opkind.as_string())
    }
}

impl Node {
    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, super::NodeType>,
        scales: &VarScales,
        idx: usize,
        symbol_values: &SymbolValues,
        run_args: &crate::RunArgs,
    ) -> Result<Self, GraphError> {
        trace!("Create {:?}", node);
        trace!("Create op {:?}", node.op);

        let num_uses = std::cmp::max(
            node.outputs
                .iter()
                .map(|outlet| outlet.successors.len())
                .sum::<usize>(),
            1,
        );

        let mut inputs = Vec::with_capacity(node.inputs.len());
        let mut input_ids = Vec::with_capacity(node.inputs.len());

        for i in &node.inputs {
            let input_node = other_nodes
                .get(&i.node)
                .ok_or(GraphError::MissingInput(idx))?
                .clone();
            inputs.push(input_node);
            input_ids.push((i.node, i.slot));
        }

        let (mut opkind, deleted_indices) = new_op_from_onnx(
            idx,
            scales,
            node.clone(),
            &mut inputs,
            symbol_values,
            run_args,
        )?;

        for (i, (idx, _)) in input_ids.iter_mut().enumerate() {
            if deleted_indices.contains(&i) {
                *idx = usize::MAX;
            }
        }

        input_ids.retain(|(idx, _)| *idx != usize::MAX);

        let mut in_scales: Vec<crate::Scale> = input_ids
            .iter()
            .map(|(idx, outlet)| {
                let idx = inputs
                    .iter()
                    .position(|x| *idx == x.idx())
                    .ok_or(GraphError::MissingInput(*idx))?;
                Ok(inputs[idx].out_scales()[*outlet])
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let homogenous_inputs = opkind.requires_homogenous_input_scales();
        for input in homogenous_inputs
            .into_iter()
            .filter(|i| !deleted_indices.contains(i))
        {
            if inputs.len() > input {
                let input_node = other_nodes
                    .get_mut(&inputs[input].idx())
                    .ok_or(GraphError::MissingInput(idx))?;
                let input_opkind = &mut input_node.opkind();
                if let Some(constant) = input_opkind.get_mutable_constant() {
                    rescale_const_with_single_use(
                        constant,
                        in_scales.clone(),
                        &run_args.param_visibility,
                        input_node.num_uses(),
                    )?;
                    input_node.replace_opkind(constant.clone_dyn().into());
                    let out_scale = input_opkind.out_scale(vec![])?;
                    input_node.bump_scale(out_scale);
                    in_scales[input] = out_scale;
                }
            }
        }

        opkind = opkind.homogenous_rescale(in_scales.clone())?.into();
        let mut out_scale = opkind.out_scale(in_scales.clone())?;
        let global_scale = scales.get_max();
        opkind = RebaseScale::rebase(opkind, global_scale, out_scale, scales.rebase_multiplier);

        out_scale = opkind.out_scale(in_scales)?;

        let out_dims = node_output_shapes(&node, symbol_values)?;
        let mut out_dims = out_dims[0].clone();

        if out_dims.is_empty() {
            out_dims = vec![1];
        }

        Ok(Node {
            idx,
            opkind,
            inputs: input_ids,
            out_dims,
            out_scale,
            num_uses,
        })
    }

    pub fn is_softmax(&self) -> bool {
        matches!(self.opkind, SupportedOp::Hybrid(HybridOp::Softmax { .. }))
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
fn rescale_const_with_single_use(
    constant: &mut Constant<Fp>,
    in_scales: Vec<crate::Scale>,
    param_visibility: &Visibility,
    num_uses: usize,
) -> Result<(), GraphError> {
    if num_uses == 1 {
        let current_scale = constant.out_scale(vec![])?;
        let scale_max = in_scales.iter().max().ok_or(GraphError::MissingScale)?;
        if scale_max > &current_scale {
            let raw_values = constant.raw_values.clone();
            constant.quantized_values =
                super::quantize_tensor(raw_values, *scale_max, param_visibility)?;
        }
    }

    Ok(())
}