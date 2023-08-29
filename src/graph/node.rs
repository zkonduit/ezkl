use super::scale_to_multiplier;
#[cfg(not(target_arch = "wasm32"))]
use super::utilities::node_output_shapes;
#[cfg(not(target_arch = "wasm32"))]
use super::VarScales;
#[cfg(not(target_arch = "wasm32"))]
use super::Visibility;
use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::circuit::Constant;
use crate::circuit::Input;
use crate::circuit::Op;
use crate::circuit::Unknown;
use crate::fieldutils::felt_to_i128;
use crate::fieldutils::i128_to_felt;
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::new_op_from_onnx;
use crate::tensor::Tensor;
use crate::tensor::TensorError;
use halo2curves::bn256::Fr as Fp;
#[cfg(not(target_arch = "wasm32"))]
use itertools::Itertools;
#[cfg(not(target_arch = "wasm32"))]
use log::trace;
use serde::Deserialize;
use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use tabled::Tabled;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::{
    self,
    prelude::{Node as OnnxNode, TypedFact, TypedOp},
};

#[cfg(not(target_arch = "wasm32"))]
fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{:?}", v)
    } else {
        String::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn display_opkind(v: &SupportedOp) -> String {
    v.as_string()
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rescaled {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// The scale of the operation's inputs.
    pub scale: Vec<(usize, u128)>,
}

impl Op<Fp> for Rescaled {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<Fp>]) -> Result<crate::circuit::ForwardResult<Fp>, TensorError> {
        if self.scale.len() != x.len() {
            return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
        }

        let mut rescaled_inputs = vec![];
        let inputs = &mut x.to_vec();
        for (i, ri) in inputs.iter_mut().enumerate() {
            let mult_tensor = Tensor::from([Fp::from(self.scale[i].1 as u64)].into_iter());
            let res = (ri.clone() * mult_tensor)?;
            rescaled_inputs.push(res);
        }
        Op::<Fp>::f(&*self.inner, &rescaled_inputs)
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }

    fn as_string(&self) -> String {
        format!("RESCALED INPUT ({})", self.inner.as_string())
    }

    fn out_scale(&self, in_scales: Vec<u32>, _g: u32) -> u32 {
        let in_scales = in_scales
            .into_iter()
            .zip(self.scale.iter())
            .map(|(a, b)| a + crate::graph::mult_to_scale(b.1 as f64))
            .collect();

        Op::<Fp>::out_scale(&*self.inner, in_scales, _g)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        vec![]
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, Box<dyn Error>> {
        if self.scale.len() != values.len() {
            return Err(Box::new(TensorError::DimMismatch(
                "rescaled inputs".to_string(),
            )));
        }

        let res =
            &crate::circuit::layouts::rescale(config, region, values[..].try_into()?, &self.scale)?
                [..];
        self.inner.layout(config, region, res)
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebaseScale {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// The scale of the operation's inputs.
    pub scale: u128,
    /// rebase multiplier
    pub rebase_multiplier: u32,
}

impl RebaseScale {
    ///
    pub fn rebase(
        inner: SupportedOp,
        original_scale: u32,
        op_out_scale: u32,
        rebase_multiplier: u32,
    ) -> SupportedOp {
        if op_out_scale > rebase_multiplier * original_scale {
            SupportedOp::RebaseScale(RebaseScale {
                inner: Box::new(inner),
                scale: scale_to_multiplier(op_out_scale - rebase_multiplier * original_scale)
                    as u128,
                rebase_multiplier,
            })
        } else {
            inner
        }
    }
}

impl Op<Fp> for RebaseScale {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<Fp>]) -> Result<crate::circuit::ForwardResult<Fp>, TensorError> {
        let mut res = Op::<Fp>::f(&*self.inner, &x)?;

        let ri = res.output.map(felt_to_i128);
        let rescaled = crate::tensor::ops::nonlinearities::const_div(&ri, self.scale as f64);
        res.output = rescaled.map(i128_to_felt);

        res.intermediate_lookups.push(ri);

        Ok(res)
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }

    fn as_string(&self) -> String {
        format!("REBASED SCALE ({})", self.inner.as_string())
    }

    fn out_scale(&self, _: Vec<u32>, _g: u32) -> u32 {
        self.rebase_multiplier * _g
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        let mut lookups = self.inner.required_lookups();
        lookups.push(LookupOp::Div {
            denom: crate::circuit::utils::F32(self.scale as f32),
        });
        lookups
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, Box<dyn Error>> {
        let original_res = self.inner.layout(config, region, values)?.unwrap();

        Ok(Some(crate::circuit::layouts::nonlinearity(
            config,
            region,
            &[original_res],
            &LookupOp::Div {
                denom: crate::circuit::utils::F32(self.scale as f32),
            },
        )?))
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// A single operation in a [crate::graph::Model].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SupportedOp {
    /// A linear operation.
    Linear(PolyOp<Fp>),
    /// A nonlinear operation.
    Nonlinear(LookupOp),
    /// A hybrid operation.
    Hybrid(HybridOp),
    ///
    Input(Input),
    ///
    Constant(Constant<Fp>),
    ///
    Unknown(Unknown),
    ///
    Rescaled(Rescaled),
    ///
    RebaseScale(RebaseScale),
}

impl SupportedOp {
    ///
    pub fn get_input(&self) -> Option<Input> {
        match self {
            SupportedOp::Input(op) => Some(op.clone()),
            _ => None,
        }
    }
}

impl From<Box<dyn Op<Fp>>> for SupportedOp {
    fn from(value: Box<dyn Op<Fp>>) -> Self {
        if let Some(op) = value.as_any().downcast_ref::<PolyOp<Fp>>() {
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

        panic!("Unsupported op type")
    }
}

impl Op<Fp> for SupportedOp {
    fn f(
        &self,
        inputs: &[Tensor<Fp>],
    ) -> Result<crate::circuit::ForwardResult<Fp>, crate::tensor::TensorError> {
        match self {
            SupportedOp::Linear(op) => op.f(inputs),
            SupportedOp::Nonlinear(op) => op.f(inputs),
            SupportedOp::Hybrid(op) => op.f(inputs),
            SupportedOp::Input(op) => op.f(inputs),
            SupportedOp::Constant(op) => op.f(inputs),
            SupportedOp::Unknown(op) => op.f(inputs),
            SupportedOp::Rescaled(op) => op.f(inputs),
            SupportedOp::RebaseScale(op) => op.f(inputs),
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<Fp>,
        region: &mut crate::circuit::region::RegionCtx<Fp>,
        values: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Option<crate::tensor::ValTensor<Fp>>, Box<dyn Error>> {
        match self {
            SupportedOp::Linear(op) => op.layout(config, region, values),
            SupportedOp::Nonlinear(op) => op.layout(config, region, values),
            SupportedOp::Hybrid(op) => op.layout(config, region, values),
            SupportedOp::Input(op) => op.layout(config, region, values),
            SupportedOp::Constant(op) => op.layout(config, region, values),
            SupportedOp::Unknown(op) => op.layout(config, region, values),
            SupportedOp::Rescaled(op) => op.layout(config, region, values),
            SupportedOp::RebaseScale(op) => op.layout(config, region, values),
        }
    }

    fn is_input(&self) -> bool {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::is_input(op),
            SupportedOp::Nonlinear(op) => Op::<Fp>::is_input(op),
            SupportedOp::Hybrid(op) => Op::<Fp>::is_input(op),
            SupportedOp::Input(op) => Op::<Fp>::is_input(op),
            SupportedOp::Constant(op) => Op::<Fp>::is_input(op),
            SupportedOp::Unknown(op) => Op::<Fp>::is_input(op),
            SupportedOp::Rescaled(op) => Op::<Fp>::is_input(op),
            SupportedOp::RebaseScale(op) => Op::<Fp>::is_input(op),
        }
    }

    fn is_constant(&self) -> bool {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Nonlinear(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Hybrid(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Input(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Constant(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Unknown(op) => Op::<Fp>::is_constant(op),
            SupportedOp::Rescaled(op) => Op::<Fp>::is_constant(op),
            SupportedOp::RebaseScale(op) => Op::<Fp>::is_constant(op),
        }
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Nonlinear(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Hybrid(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Input(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Constant(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Unknown(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::Rescaled(op) => Op::<Fp>::requires_homogenous_input_scales(op),
            SupportedOp::RebaseScale(op) => Op::<Fp>::requires_homogenous_input_scales(op),
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        match self {
            SupportedOp::Linear(op) => Box::new(op.clone()),
            SupportedOp::Nonlinear(op) => Box::new(op.clone()),
            SupportedOp::Hybrid(op) => Box::new(op.clone()),
            SupportedOp::Input(op) => Box::new(op.clone()),
            SupportedOp::Constant(op) => Box::new(op.clone()),
            SupportedOp::Unknown(op) => Box::new(op.clone()),
            SupportedOp::Rescaled(op) => Box::new(op.clone()),
            SupportedOp::RebaseScale(op) => Box::new(op.clone()),
        }
    }

    fn as_string(&self) -> String {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::as_string(op),
            SupportedOp::Nonlinear(op) => Op::<Fp>::as_string(op),
            SupportedOp::Hybrid(op) => Op::<Fp>::as_string(op),
            SupportedOp::Input(op) => Op::<Fp>::as_string(op),
            SupportedOp::Constant(op) => Op::<Fp>::as_string(op),
            SupportedOp::Unknown(op) => Op::<Fp>::as_string(op),
            SupportedOp::Rescaled(op) => Op::<Fp>::as_string(op),
            SupportedOp::RebaseScale(op) => Op::<Fp>::as_string(op),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Nonlinear(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Hybrid(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Input(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Constant(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Unknown(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::Rescaled(op) => Op::<Fp>::required_lookups(op),
            SupportedOp::RebaseScale(op) => Op::<Fp>::required_lookups(op),
        }
    }

    fn rescale(&self, in_scales: Vec<u32>, out_scale: u32) -> Box<dyn Op<Fp>> {
        match self {
            SupportedOp::Linear(op) => {
                let inputs_to_scale = self.requires_homogenous_input_scales();
                // creates a rescaled op if the inputs are not homogenous
                super::homogenize_input_scales(Box::new(op.clone()), in_scales, inputs_to_scale)
                    .unwrap()
            }
            SupportedOp::Nonlinear(op) => op.rescale(in_scales, out_scale),
            SupportedOp::Hybrid(op) => op.rescale(in_scales, out_scale),
            SupportedOp::Input(op) => op.rescale(in_scales, out_scale),
            SupportedOp::Constant(op) => op.rescale(in_scales, out_scale),
            SupportedOp::Unknown(op) => op.rescale(in_scales, out_scale),
            SupportedOp::Rescaled(op) => op.rescale(in_scales, out_scale),
            SupportedOp::RebaseScale(op) => op.rescale(in_scales, out_scale),
        }
    }

    fn out_scale(&self, in_scales: Vec<u32>, global: u32) -> u32 {
        match self {
            SupportedOp::Linear(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Nonlinear(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Hybrid(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Input(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Constant(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Unknown(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::Rescaled(op) => Op::<Fp>::out_scale(op, in_scales, global),
            SupportedOp::RebaseScale(op) => Op::<Fp>::out_scale(op, in_scales, global),
        }
    }
}

/// A node's input is a tensor from another node's output.
pub type Outlet = (usize, usize);

/// A single operation in a [crate::graph::Model].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// [Op] i.e what operation this node represents.
    pub opkind: SupportedOp,
    /// The denominator in the fixed point representation for the node's output. Tensors of differing scales should not be combined.
    pub out_scale: u32,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    /// The indices of the node's inputs.
    pub inputs: Vec<Outlet>,
    /// Dimensions of output.
    pub out_dims: Vec<usize>,
    /// The node's unique identifier.
    pub idx: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Tabled for Node {
    const LENGTH: usize = 6;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        let mut headers = Vec::with_capacity(Self::LENGTH);
        for i in [
            "idx",
            "opkind",
            "out_scale",
            "inputs",
            "out_dims",
            "required_lookups",
        ] {
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
        fields.push(std::borrow::Cow::Owned(format!(
            "{:?}",
            self.opkind
                .required_lookups()
                .iter()
                .map(<LookupOp as Op<Fp>>::as_string)
                .collect_vec()
        )));
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
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the computational graph.
    /// * `public_params` - flag if parameters of model are public
    /// * `idx` - The node's unique identifier.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, super::NodeType>,
        scales: &VarScales,
        param_visibility: Visibility,
        idx: usize,
    ) -> Result<Self, Box<dyn Error>> {
        trace!("Create {:?}", node);
        trace!("Create op {:?}", node.op);

        // load the node inputs
        let mut inputs = vec![];

        // we can only take the inputs as mutable once -- so we need to collect them first
        let mut input_ids = node
            .inputs
            .iter()
            .map(|i| (i.node, i.slot))
            .collect::<Vec<_>>();

        input_ids.iter().for_each(|(i, _)| {
            inputs.push(other_nodes.get(i).ok_or("input not found").unwrap().clone())
        });

        let (mut opkind, deleted_indices) =
            new_op_from_onnx(idx, scales, param_visibility, node.clone(), &mut inputs)?; // parses the op name

        // we can only take the inputs as mutable once -- so we need to collect them first
        other_nodes.extend(
            inputs
                .iter()
                .map(|i| (i.idx(), i.clone()))
                .collect::<BTreeMap<_, _>>(),
        );

        input_ids.iter_mut().enumerate().for_each(|(i, (idx, _))| {
            if deleted_indices.contains(&i) {
                // this input is not used
                *idx = usize::MAX;
            }
        });

        // remove the inputs that are not used
        input_ids.retain(|(idx, _)| *idx != usize::MAX);

        // rescale the inputs if necessary to get consistent fixed points
        let in_scales: Vec<u32> = input_ids
            .iter()
            .map(|(idx, outlet)| {
                let idx = inputs.iter().position(|x| *idx == x.idx()).unwrap();
                inputs[idx].out_scales()[*outlet]
            })
            .collect();

        opkind = opkind.rescale(in_scales.clone(), scales.input).into();
        let mut out_scale = opkind.out_scale(in_scales.clone(), scales.input);
        opkind = RebaseScale::rebase(opkind, scales.input, out_scale, scales.rebase_multiplier);
        out_scale = opkind.out_scale(in_scales, scales.input);

        // get the output shape
        let out_dims = {
            let output_shapes = match node_output_shapes(&node) {
                Ok(s) => Some(s),
                _ => None,
            };

            if let Some([Some(v)]) = output_shapes.as_deref() {
                v.to_vec()
            } else {
                panic!("Could not get output shape for node {:?}", node);
            }
        };

        Ok(Node {
            idx,
            opkind,
            inputs: input_ids,
            out_dims,
            out_scale,
        })
    }
}
