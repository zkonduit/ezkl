//! Dequantize ONNX models that have been mangled by post-training
//! quantization.
//!
//! `onnxruntime.quantization` (and similar tools) rewrite a float ONNX model
//! to insert quantize / dequantize / integer-arithmetic operators before
//! export. Tract — the ONNX parser EZKL relies on — cannot analyse those
//! patterns, and EZKL itself already quantizes internally via its `scale`
//! run-arg, so a pre-quantized model is both redundant and unsupported.
//!
//! This module canonicalises a [`tract_onnx::pb::ModelProto`] in place,
//! collapsing the quantization patterns we know about back into their float
//! equivalents. The single entry point is [`apply`]; it is invoked
//! automatically from [`crate::graph::Model::new`] during ONNX loading, and
//! the [`crate::commands::Commands::Dequantize`] CLI subcommand exposes the
//! same logic so users can persist a cleaned model.
//!
//! # Patterns handled
//!
//! 1. **Activation Q/DQ identity pair.** A `QuantizeLinear` immediately
//!    followed by a `DequantizeLinear` that share the same `scale` and
//!    `zero_point` is mathematically the identity (modulo quantization
//!    noise). Both nodes are removed and consumers are rewired to the input
//!    of the `QuantizeLinear`.
//!
//! 2. **Standalone weight `DequantizeLinear`.** When a weight initializer is
//!    quantized at export time and dequantized inline before use, the entire
//!    `DequantizeLinear(W_int, scale, zp)` is folded into a single float
//!    initializer `W_float = (W_int - zp) * scale`. Consumers are rewired to
//!    the new initializer.
//!
//! 3. **`DynamicQuantizeLinear` + integer-op fusion.** This is the pattern
//!    that `quantize_dynamic` emits around each `Conv` / `MatMul` it
//!    rewrites:
//!
//!    ```text
//!    x ──► DynamicQuantizeLinear ──► [x_q, x_scale, x_zp]
//!    Mul(x_scale, W_scale) ──► combined_scale
//!    IntegerOp(x_q, W_q, x_zp, W_zp) ──► y_int
//!    Cast(y_int) ──► y_f32
//!    Mul(y_f32, combined_scale) ──► y
//!    ```
//!
//!    The five-node subgraph is replaced by a single `MatMul` / `Conv` over
//!    `(x, (W_q - W_zp) * W_scale)`. Spatial attributes on `ConvInteger` are
//!    preserved on the replacement `Conv`. Trailing bias `Add` nodes are
//!    left untouched.
//!
//! Patterns we do **not** rewrite (`QLinearConv`, `QLinearMatMul`,
//! `QLinearAdd`, etc.) are left in the graph; the safety-net detector in
//! [`crate::graph::model::Model::reject_onnx_quantization_ops`] reports them
//! with an actionable error.

#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use std::collections::{HashMap, HashSet};

use tract_onnx::pb::tensor_proto::DataType;
use tract_onnx::pb::{ModelProto, NodeProto, TensorProto};

/// Outcome of a single [`apply`] invocation. Counts are exposed both for
/// logging by `Model::new` and for the `ezkl dequantize` subcommand to
/// report what changed.
#[derive(Debug, Default, Clone)]
pub struct DequantizationReport {
    /// Number of `QuantizeLinear -> DequantizeLinear` identity pairs folded.
    pub qdq_pairs_collapsed: usize,
    /// Number of standalone weight `DequantizeLinear` nodes folded into
    /// float initializers.
    pub weight_dq_folded: usize,
    /// Number of `DynamicQuantizeLinear + integer-op` fusions rewritten to
    /// plain `MatMul` / `Conv` nodes.
    pub dyn_quant_fusions_replaced: usize,
    /// Quantization-related operators that survived the rewrite, formatted
    /// as `node_name (OpType)`. A non-empty list does not by itself signal
    /// failure — the detector in `Model::new` decides whether to error.
    pub remaining_quantization_ops: Vec<String>,
}

impl DequantizationReport {
    /// True when at least one rewrite was performed.
    pub fn changed(&self) -> bool {
        self.qdq_pairs_collapsed > 0
            || self.weight_dq_folded > 0
            || self.dyn_quant_fusions_replaced > 0
    }

    /// True when every quantization-related operator has been resolved.
    pub fn is_clean(&self) -> bool {
        self.remaining_quantization_ops.is_empty()
    }
}

/// Errors specific to the dequantize pass.
#[derive(Debug, thiserror::Error)]
pub enum DequantizationError {
    /// The model contained an initializer with a data type the pass doesn't
    /// know how to read (e.g. complex, string, fp16).
    #[error("unsupported tensor dtype while reading initializer `{name}`: {dtype}")]
    UnsupportedTensorDtype {
        /// Name of the offending initializer in the ONNX graph.
        name: String,
        /// Numeric `TensorProto.data_type` value we couldn't decode.
        dtype: i32,
    },
    /// Dequantizing a weight requires the int weight, scale, and zero-point
    /// element counts to broadcast cleanly. We only handle the per-tensor
    /// (scalar scale/zp) case today; per-channel quantization would need
    /// shape-aware broadcasting.
    #[error(
        "cannot dequantize weight `{weight}`: scale has {scale_len} elements and zp has {zp_len}; \
         only per-tensor (scalar) parameters are supported"
    )]
    UnsupportedQuantParamShape {
        /// Name of the weight initializer being dequantized.
        weight: String,
        /// Element count of the scale tensor (expected 1 for per-tensor).
        scale_len: usize,
        /// Element count of the zero-point tensor (expected 1 for per-tensor).
        zp_len: usize,
    },
}

const QUANT_OP_PREFIXES: &[&str] = &[
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinear",
];
const QUANT_OP_NAMES: &[&str] = &["ConvInteger", "MatMulInteger"];

fn is_quantization_op(op_type: &str) -> bool {
    QUANT_OP_NAMES.iter().any(|n| *n == op_type)
        || QUANT_OP_PREFIXES.iter().any(|p| op_type.starts_with(p))
}

// ---------------------------------------------------------------------------
// Tensor helpers — read TensorProto data into f32 / i32 vectors regardless of
// whether the bytes live in the typed `*_data` fields or in `raw_data`.
// ---------------------------------------------------------------------------

fn read_le<T: Copy + Default, const N: usize>(bytes: &[u8], cvt: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|c| {
            let mut buf = [0u8; N];
            buf.copy_from_slice(c);
            cvt(buf)
        })
        .collect()
}

/// Read `t` as a flat `Vec<f32>`, regardless of the storage variant ONNX
/// chose for it. Returns an error for tensor data types we cannot represent
/// as f32 in a meaningful way (string, complex, etc.).
fn tensor_to_f32(t: &TensorProto) -> Result<Vec<f32>, DequantizationError> {
    let dtype = DataType::from_i32(t.data_type).unwrap_or(DataType::Undefined);
    let from_raw_or = |typed: Vec<f32>, raw: Vec<f32>| -> Vec<f32> {
        if !typed.is_empty() {
            typed
        } else {
            raw
        }
    };
    Ok(match dtype {
        DataType::Float => from_raw_or(
            t.float_data.clone(),
            read_le::<f32, 4>(&t.raw_data, f32::from_le_bytes),
        ),
        DataType::Int8 => {
            if !t.int32_data.is_empty() {
                t.int32_data.iter().map(|v| *v as f32).collect()
            } else {
                t.raw_data.iter().map(|b| (*b as i8) as f32).collect()
            }
        }
        DataType::Uint8 => {
            if !t.int32_data.is_empty() {
                t.int32_data.iter().map(|v| *v as f32).collect()
            } else {
                t.raw_data.iter().map(|b| *b as f32).collect()
            }
        }
        DataType::Int16 => {
            if !t.int32_data.is_empty() {
                t.int32_data.iter().map(|v| *v as f32).collect()
            } else {
                read_le::<i16, 2>(&t.raw_data, i16::from_le_bytes)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        }
        DataType::Uint16 => {
            if !t.int32_data.is_empty() {
                t.int32_data.iter().map(|v| *v as f32).collect()
            } else {
                read_le::<u16, 2>(&t.raw_data, u16::from_le_bytes)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        }
        DataType::Int32 => {
            if !t.int32_data.is_empty() {
                t.int32_data.iter().map(|v| *v as f32).collect()
            } else {
                read_le::<i32, 4>(&t.raw_data, i32::from_le_bytes)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        }
        DataType::Int64 => {
            if !t.int64_data.is_empty() {
                t.int64_data.iter().map(|v| *v as f32).collect()
            } else {
                read_le::<i64, 8>(&t.raw_data, i64::from_le_bytes)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        }
        DataType::Double => {
            if !t.double_data.is_empty() {
                t.double_data.iter().map(|v| *v as f32).collect()
            } else {
                read_le::<f64, 8>(&t.raw_data, f64::from_le_bytes)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        }
        _ => {
            return Err(DequantizationError::UnsupportedTensorDtype {
                name: t.name.clone(),
                dtype: t.data_type,
            });
        }
    })
}

/// Pack `values` into a `TensorProto` of dtype FLOAT, storing the bytes in
/// `raw_data` (the modern ONNX convention for sizeable tensors). `dims`
/// preserves the original shape.
fn make_float_initializer(name: String, dims: Vec<i64>, values: &[f32]) -> TensorProto {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    TensorProto {
        dims,
        data_type: DataType::Float as i32,
        raw_data: bytes,
        name,
        ..TensorProto::default()
    }
}

/// Apply `(W_int - zp) * scale` element-wise to obtain the float weight.
/// Currently only supports per-tensor (scalar) `scale` and `zp`; per-channel
/// would require knowing the quantization axis.
fn dequantize_weight(
    w_int: &TensorProto,
    scale: &TensorProto,
    zero_point: Option<&TensorProto>,
) -> Result<Vec<f32>, DequantizationError> {
    let scale_vals = tensor_to_f32(scale)?;
    let zp_vals = match zero_point {
        Some(zp) => tensor_to_f32(zp)?,
        None => vec![0.0],
    };
    if scale_vals.len() != 1 || zp_vals.len() != 1 {
        return Err(DequantizationError::UnsupportedQuantParamShape {
            weight: w_int.name.clone(),
            scale_len: scale_vals.len(),
            zp_len: zp_vals.len(),
        });
    }
    let s = scale_vals[0];
    let zp = zp_vals[0];
    let w_int_vals = tensor_to_f32(w_int)?;
    Ok(w_int_vals.into_iter().map(|v| (v - zp) * s).collect())
}

// ---------------------------------------------------------------------------
// Pattern 1: collapse Q -> DQ identity pairs on activations.
// ---------------------------------------------------------------------------

fn fold_qdq_identity_pairs(
    nodes: Vec<NodeProto>,
    _inits: &mut HashMap<String, TensorProto>,
    report: &mut DequantizationReport,
) -> Vec<NodeProto> {
    // Build an output -> producer index so we can locate the Q feeding each DQ
    // without an O(n^2) search.
    let mut producer_of: HashMap<&str, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for o in &n.output {
            producer_of.insert(o.as_str(), i);
        }
    }

    let mut drop = HashSet::<usize>::new();
    let mut rewire = HashMap::<String, String>::new();

    for (dq_idx, dq) in nodes.iter().enumerate() {
        if dq.op_type != "DequantizeLinear" {
            continue;
        }
        let q_input = match dq.input.first() {
            Some(s) => s.as_str(),
            None => continue,
        };
        let q_idx = match producer_of.get(q_input) {
            Some(i) => *i,
            None => continue,
        };
        let q = &nodes[q_idx];
        if q.op_type != "QuantizeLinear" {
            continue;
        }
        // Both nodes must reference the same scale + zero_point tensors.
        if q.input.get(1..) != dq.input.get(1..) {
            continue;
        }
        // Reroute the DQ output back to the Q input; mark both nodes for drop.
        rewire.insert(dq.output[0].clone(), q.input[0].clone());
        drop.insert(q_idx);
        drop.insert(dq_idx);
        report.qdq_pairs_collapsed += 1;
    }

    apply_drops_and_rewires(nodes, &drop, &rewire)
}

fn apply_drops_and_rewires(
    nodes: Vec<NodeProto>,
    drop: &HashSet<usize>,
    rewire: &HashMap<String, String>,
) -> Vec<NodeProto> {
    nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut n)| {
            if drop.contains(&i) {
                return None;
            }
            for inp in n.input.iter_mut() {
                if let Some(new) = rewire.get(inp.as_str()) {
                    *inp = new.clone();
                }
            }
            Some(n)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pattern 2: fold standalone weight DequantizeLinear nodes into float inits.
// ---------------------------------------------------------------------------

fn fold_weight_dequantize_linear(
    nodes: Vec<NodeProto>,
    inits: &mut HashMap<String, TensorProto>,
    report: &mut DequantizationReport,
) -> Result<Vec<NodeProto>, DequantizationError> {
    let mut drop = HashSet::<usize>::new();
    let mut rewire = HashMap::<String, String>::new();
    let mut new_inits = Vec::<TensorProto>::new();

    for (idx, n) in nodes.iter().enumerate() {
        if n.op_type != "DequantizeLinear" {
            continue;
        }
        if n.input.is_empty() {
            continue;
        }
        let w_name = n.input[0].as_str();
        let w_init = match inits.get(w_name) {
            Some(t) => t,
            None => continue, // activation DQ — handled by pattern 1
        };
        let scale_name = match n.input.get(1) {
            Some(s) => s.as_str(),
            None => continue,
        };
        let scale_init = match inits.get(scale_name) {
            Some(t) => t,
            None => continue,
        };
        let zp_init = n
            .input
            .get(2)
            .and_then(|name| inits.get(name.as_str()))
            .cloned();

        let dequant = dequantize_weight(w_init, scale_init, zp_init.as_ref())?;
        let folded_name = format!("{}__dequantized", w_name);
        let init = make_float_initializer(folded_name.clone(), w_init.dims.clone(), &dequant);
        new_inits.push(init);
        rewire.insert(n.output[0].clone(), folded_name);
        drop.insert(idx);
        report.weight_dq_folded += 1;
    }

    for ini in new_inits {
        inits.insert(ini.name.clone(), ini);
    }
    Ok(apply_drops_and_rewires(nodes, &drop, &rewire))
}

// ---------------------------------------------------------------------------
// Pattern 3: collapse DynamicQuantizeLinear + IntegerOp + Cast + Mul fusion
// emitted by `quantize_dynamic` for each rewritten MatMul / Conv.
// ---------------------------------------------------------------------------

fn copy_attributes(src: &NodeProto, dst: &mut NodeProto) {
    dst.attribute = src.attribute.clone();
}

fn collapse_dynamic_quantize_fusion(
    nodes: Vec<NodeProto>,
    inits: &mut HashMap<String, TensorProto>,
    report: &mut DequantizationReport,
) -> Result<Vec<NodeProto>, DequantizationError> {
    // Index nodes by the tensors they produce / consume so we can walk the
    // chain without quadratic scans. `producer` holds (node_index) per output
    // tensor name; `consumers_of` holds the indices of every node consuming a
    // given tensor. These indices remain valid throughout this pass: we never
    // mutate `nodes` until the very end.
    let mut producer: HashMap<&str, usize> = HashMap::new();
    let mut consumers_of: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for o in &n.output {
            producer.insert(o.as_str(), i);
        }
        for inp in &n.input {
            consumers_of.entry(inp.as_str()).or_default().push(i);
        }
    }

    let mut drop = HashSet::<usize>::new();
    let mut replacements: HashMap<usize, NodeProto> = HashMap::new();
    let mut new_inits: Vec<TensorProto> = Vec::new();

    for (int_idx, int_op) in nodes.iter().enumerate() {
        let replacement_op_type = match int_op.op_type.as_str() {
            "MatMulInteger" => "MatMul",
            "ConvInteger" => "Conv",
            _ => continue,
        };

        // input[0] of the integer op must be the first output of a
        // DynamicQuantizeLinear; this gives us x (the original f32 input).
        let x_q_name = match int_op.input.first() {
            Some(s) => s.as_str(),
            None => continue,
        };
        let dq_idx = match producer.get(x_q_name) {
            Some(i) => *i,
            None => continue,
        };
        let dq_node = &nodes[dq_idx];
        if dq_node.op_type != "DynamicQuantizeLinear" {
            continue;
        }
        let x_input = dq_node.input[0].clone();
        let x_scale_name = dq_node.output.get(1).cloned().unwrap_or_default();
        let x_zp_name = dq_node.output.get(2).cloned().unwrap_or_default();
        if x_zp_name.is_empty() || int_op.input.get(2).map(String::as_str) != Some(&x_zp_name) {
            continue;
        }

        // Weight + weight zp must be initializers (per-tensor quantization).
        let w_name = match int_op.input.get(1) {
            Some(s) => s.as_str(),
            None => continue,
        };
        if !inits.contains_key(w_name) {
            continue;
        }
        let w_zp_name = int_op.input.get(3).cloned();
        if let Some(name) = &w_zp_name {
            if !inits.contains_key(name.as_str()) {
                continue;
            }
        }

        // Walk forward: int_op output -> Cast -> Mul(combined_scale).
        let int_out_name = int_op.output[0].as_str();
        let cast_idx = match consumers_of.get(int_out_name).and_then(|cs| {
            cs.iter()
                .copied()
                .find(|i| nodes[*i].op_type == "Cast")
        }) {
            Some(i) => i,
            None => continue,
        };
        let cast_node = &nodes[cast_idx];
        let cast_out_name = cast_node.output[0].as_str();
        let out_mul_idx = match consumers_of.get(cast_out_name).and_then(|cs| {
            cs.iter()
                .copied()
                .find(|i| nodes[*i].op_type == "Mul")
        }) {
            Some(i) => i,
            None => continue,
        };
        let out_mul_node = &nodes[out_mul_idx];
        let combined_scale_name = if out_mul_node.input[0] == cast_out_name {
            out_mul_node.input.get(1).cloned()
        } else {
            out_mul_node.input.first().cloned()
        };
        let combined_scale_name = match combined_scale_name {
            Some(s) => s,
            None => continue,
        };
        let final_out_name = out_mul_node.output[0].clone();

        // Walk back: combined_scale must be the output of Mul(x_scale, W_scale)
        // where W_scale is an initializer.
        let scale_mul_idx = match producer.get(combined_scale_name.as_str()) {
            Some(i) => *i,
            None => continue,
        };
        let scale_mul_node = &nodes[scale_mul_idx];
        if scale_mul_node.op_type != "Mul" {
            continue;
        }
        let scale_inputs = &scale_mul_node.input;
        if !scale_inputs.iter().any(|n| n == &x_scale_name) {
            continue;
        }
        let w_scale_name = if scale_inputs[0] == x_scale_name {
            scale_inputs.get(1).cloned()
        } else {
            scale_inputs.first().cloned()
        };
        let w_scale_name = match w_scale_name {
            Some(s) => s,
            None => continue,
        };
        if !inits.contains_key(w_scale_name.as_str()) {
            continue;
        }

        // All checks passed — synthesise the dequantized weight and the
        // replacement Conv / MatMul node.
        let w_init = inits[w_name].clone();
        let scale_init = inits[w_scale_name.as_str()].clone();
        let zp_init = w_zp_name
            .as_deref()
            .and_then(|n| inits.get(n))
            .cloned();
        let w_float = dequantize_weight(&w_init, &scale_init, zp_init.as_ref())?;
        let float_w_name = format!("{}__dequantized", w_name);
        new_inits.push(make_float_initializer(
            float_w_name.clone(),
            w_init.dims.clone(),
            &w_float,
        ));

        let mut replacement = NodeProto {
            input: vec![x_input, float_w_name],
            output: vec![final_out_name],
            name: format!(
                "{}__dequantized",
                if int_op.name.is_empty() {
                    replacement_op_type
                } else {
                    int_op.name.as_str()
                }
            ),
            op_type: replacement_op_type.to_string(),
            ..NodeProto::default()
        };
        if replacement_op_type == "Conv" {
            copy_attributes(int_op, &mut replacement);
        }
        replacements.insert(int_idx, replacement);

        // Mark the entire subgraph for removal. The DynamicQuantizeLinear may
        // be re-marked across iterations when shared between several integer
        // ops; that's fine (sets are idempotent).
        drop.extend([dq_idx, scale_mul_idx, int_idx, cast_idx, out_mul_idx]);
        report.dyn_quant_fusions_replaced += 1;
    }

    if replacements.is_empty() {
        return Ok(nodes);
    }

    for ini in new_inits {
        inits.insert(ini.name.clone(), ini);
    }

    // Insert each replacement at the position of the integer op it replaces
    // so the resulting node list stays topologically sorted.
    let mut rebuilt = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.into_iter().enumerate() {
        if let Some(rep) = replacements.remove(&i) {
            rebuilt.push(rep);
        } else if !drop.contains(&i) {
            rebuilt.push(n);
        }
    }
    Ok(rebuilt)
}

// ---------------------------------------------------------------------------
// Cleanup + top-level entry point.
// ---------------------------------------------------------------------------

fn drop_orphan_initializers(
    nodes: &[NodeProto],
    graph_outputs: &[String],
    inits: HashMap<String, TensorProto>,
) -> HashMap<String, TensorProto> {
    let mut used: HashSet<&str> = graph_outputs.iter().map(|s| s.as_str()).collect();
    for n in nodes {
        for inp in &n.input {
            used.insert(inp.as_str());
        }
    }
    inits
        .into_iter()
        .filter(|(k, _)| used.contains(k.as_str()))
        .collect()
}

fn collect_remaining_quant_ops(nodes: &[NodeProto]) -> Vec<String> {
    nodes
        .iter()
        .filter(|n| is_quantization_op(&n.op_type))
        .map(|n| {
            let nm = if n.name.is_empty() {
                "<unnamed>"
            } else {
                n.name.as_str()
            };
            format!("{} ({})", nm, n.op_type)
        })
        .collect()
}

/// Canonicalise `model` in place, collapsing every supported PTQ pattern.
///
/// Returns a [`DequantizationReport`] summarising what changed and which (if any)
/// quantization ops survived. A non-empty
/// [`DequantizationReport::remaining_quantization_ops`] is *informational* — callers
/// decide whether to surface it as an error (the loader does so via
/// [`crate::graph::errors::GraphError::UnsupportedQuantizationOps`]; the
/// `dequantize` subcommand prints a warning).
pub fn apply(model: &mut ModelProto) -> Result<DequantizationReport, DequantizationError> {
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return Ok(DequantizationReport::default()),
    };

    let mut report = DequantizationReport::default();
    let mut nodes = std::mem::take(&mut graph.node);
    let mut inits: HashMap<String, TensorProto> = std::mem::take(&mut graph.initializer)
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();

    // Pattern 3 first: it dissolves DynamicQuantizeLinear sources that the
    // other passes wouldn't recognise. The remaining passes then sweep up any
    // standalone Q/DQ that survived.
    nodes = collapse_dynamic_quantize_fusion(nodes, &mut inits, &mut report)?;
    nodes = fold_qdq_identity_pairs(nodes, &mut inits, &mut report);
    nodes = fold_weight_dequantize_linear(nodes, &mut inits, &mut report)?;

    let graph_outputs: Vec<String> = graph.output.iter().map(|v| v.name.clone()).collect();
    let inits = drop_orphan_initializers(&nodes, &graph_outputs, inits);

    report.remaining_quantization_ops = collect_remaining_quant_ops(&nodes);

    graph.node = nodes;
    graph.initializer = inits.into_values().collect();

    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::pb::tensor_proto::DataType;
    use tract_onnx::pb::{AttributeProto, GraphProto, ValueInfoProto};

    fn fresh_model(graph: GraphProto) -> ModelProto {
        ModelProto {
            graph: Some(graph),
            ..ModelProto::default()
        }
    }

    fn vinfo(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..ValueInfoProto::default()
        }
    }

    fn scalar_f32(name: &str, v: f32) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Float as i32,
            dims: vec![],
            float_data: vec![v],
            ..TensorProto::default()
        }
    }

    fn scalar_i8(name: &str, v: i8) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Int8 as i32,
            dims: vec![],
            raw_data: vec![v as u8],
            ..TensorProto::default()
        }
    }

    fn weight_i8(name: &str, dims: Vec<i64>, vals: &[i8]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Int8 as i32,
            dims,
            raw_data: vals.iter().map(|v| *v as u8).collect(),
            ..TensorProto::default()
        }
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: name.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..NodeProto::default()
        }
    }

    fn node_with_attr(
        op: &str,
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        attr: AttributeProto,
    ) -> NodeProto {
        let mut n = node(op, name, inputs, outputs);
        n.attribute.push(attr);
        n
    }

    fn ints_attr(name: &str, vals: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: 7, // INTS
            ints: vals.to_vec(),
            ..AttributeProto::default()
        }
    }

    #[test]
    fn fold_qdq_identity_pair_collapses_to_direct_edge() {
        let graph = GraphProto {
            node: vec![
                node("QuantizeLinear", "q", &["x", "s", "zp"], &["x_q"]),
                node("DequantizeLinear", "dq", &["x_q", "s", "zp"], &["x_dq"]),
                node("Relu", "r", &["x_dq"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![scalar_f32("s", 0.1), scalar_i8("zp", 0)],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert_eq!(r.qdq_pairs_collapsed, 1);
        let g = m.graph.unwrap();
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "Relu");
        assert_eq!(g.node[0].input, vec!["x".to_string()]);
        assert!(r.is_clean());
    }

    #[test]
    fn weight_dequantize_linear_is_folded_into_float_initializer() {
        let graph = GraphProto {
            node: vec![
                node(
                    "DequantizeLinear",
                    "dq",
                    &["W_int", "W_scale", "W_zp"],
                    &["W_float"],
                ),
                node("MatMul", "mm", &["x", "W_float"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![
                weight_i8("W_int", vec![2, 2], &[10, -10, 20, -20]),
                scalar_f32("W_scale", 0.5),
                scalar_i8("W_zp", 0),
            ],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert_eq!(r.weight_dq_folded, 1);
        let g = m.graph.unwrap();
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "MatMul");
        let folded = g
            .initializer
            .iter()
            .find(|t| t.name == "W_int__dequantized")
            .expect("folded initializer missing");
        let folded_vals = tensor_to_f32(folded).unwrap();
        assert_eq!(folded_vals, vec![5.0, -5.0, 10.0, -10.0]);
        assert!(r.is_clean());
    }

    #[test]
    fn dynamic_quantize_matmul_integer_fusion_collapses_to_matmul() {
        let graph = GraphProto {
            node: vec![
                node(
                    "DynamicQuantizeLinear",
                    "dql",
                    &["x"],
                    &["x_q", "x_scale", "x_zp"],
                ),
                node("Mul", "scale_mul", &["x_scale", "W_scale"], &["combined"]),
                node(
                    "MatMulInteger",
                    "mmi",
                    &["x_q", "W_q", "x_zp", "W_zp"],
                    &["y_int"],
                ),
                node("Cast", "cast", &["y_int"], &["y_f32"]),
                node("Mul", "out_mul", &["y_f32", "combined"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![
                weight_i8("W_q", vec![2, 2], &[10, -10, 20, -20]),
                scalar_f32("W_scale", 0.5),
                scalar_i8("W_zp", 0),
            ],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert_eq!(r.dyn_quant_fusions_replaced, 1);
        let g = m.graph.unwrap();
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "MatMul");
        assert_eq!(g.node[0].input, vec!["x".to_string(), "W_q__dequantized".to_string()]);
        assert_eq!(g.node[0].output, vec!["y".to_string()]);
        assert!(r.is_clean());
    }

    #[test]
    fn dynamic_quantize_conv_integer_preserves_spatial_attributes() {
        let graph = GraphProto {
            node: vec![
                node(
                    "DynamicQuantizeLinear",
                    "dql",
                    &["x"],
                    &["x_q", "x_scale", "x_zp"],
                ),
                node("Mul", "scale_mul", &["x_scale", "W_scale"], &["combined"]),
                node_with_attr(
                    "ConvInteger",
                    "ci",
                    &["x_q", "W_q", "x_zp", "W_zp"],
                    &["y_int"],
                    ints_attr("kernel_shape", &[3, 3]),
                ),
                node("Cast", "cast", &["y_int"], &["y_f32"]),
                node("Mul", "out_mul", &["y_f32", "combined"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![
                weight_i8("W_q", vec![1, 1, 3, 3], &[1; 9]),
                scalar_f32("W_scale", 1.0),
                scalar_i8("W_zp", 0),
            ],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert_eq!(r.dyn_quant_fusions_replaced, 1);
        let g = m.graph.unwrap();
        let conv = g.node.iter().find(|n| n.op_type == "Conv").expect("Conv replacement missing");
        let kernel = conv
            .attribute
            .iter()
            .find(|a| a.name == "kernel_shape")
            .expect("kernel_shape attribute lost");
        assert_eq!(kernel.ints, vec![3, 3]);
    }

    #[test]
    fn float_only_model_is_left_unchanged() {
        let graph = GraphProto {
            node: vec![node("Relu", "r", &["x"], &["y"])],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert!(!r.changed());
        assert!(r.is_clean());
        assert_eq!(m.graph.unwrap().node.len(), 1);
    }

    #[test]
    fn idempotent_when_run_twice() {
        let graph = GraphProto {
            node: vec![
                node("QuantizeLinear", "q", &["x", "s", "zp"], &["x_q"]),
                node("DequantizeLinear", "dq", &["x_q", "s", "zp"], &["x_dq"]),
                node("Relu", "r", &["x_dq"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![scalar_f32("s", 0.1), scalar_i8("zp", 0)],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let _ = apply(&mut m).unwrap();
        let snapshot = m.clone();
        let r2 = apply(&mut m).unwrap();
        assert!(!r2.changed());
        assert_eq!(snapshot, m);
    }

    #[test]
    fn unsupported_op_is_reported_but_not_an_error() {
        let graph = GraphProto {
            node: vec![node(
                "QLinearConv",
                "qlc",
                &["x", "x_s", "x_zp", "W", "W_s", "W_zp", "y_s", "y_zp"],
                &["y"],
            )],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert!(!r.changed());
        assert!(!r.is_clean());
        assert!(r.remaining_quantization_ops.iter().any(|s| s.contains("QLinearConv")));
    }

    #[test]
    fn shared_dynamic_quantize_feeds_multiple_integer_ops() {
        // Two ConvInteger nodes sharing one DynamicQuantizeLinear — the case
        // that previously broke pattern 3 because the producer index was
        // filtered out after the first iteration.
        let graph = GraphProto {
            node: vec![
                node(
                    "DynamicQuantizeLinear",
                    "dql",
                    &["x"],
                    &["x_q", "x_scale", "x_zp"],
                ),
                node("Mul", "scale_a", &["x_scale", "W_a_scale"], &["combined_a"]),
                node(
                    "MatMulInteger",
                    "mma",
                    &["x_q", "W_a", "x_zp", "W_a_zp"],
                    &["a_int"],
                ),
                node("Cast", "cast_a", &["a_int"], &["a_f32"]),
                node("Mul", "out_a", &["a_f32", "combined_a"], &["a"]),
                node("Mul", "scale_b", &["x_scale", "W_b_scale"], &["combined_b"]),
                node(
                    "MatMulInteger",
                    "mmb",
                    &["x_q", "W_b", "x_zp", "W_b_zp"],
                    &["b_int"],
                ),
                node("Cast", "cast_b", &["b_int"], &["b_f32"]),
                node("Mul", "out_b", &["b_f32", "combined_b"], &["b"]),
                node("Add", "add", &["a", "b"], &["y"]),
            ],
            input: vec![vinfo("x")],
            output: vec![vinfo("y")],
            initializer: vec![
                weight_i8("W_a", vec![2, 2], &[1, 2, 3, 4]),
                scalar_f32("W_a_scale", 0.1),
                scalar_i8("W_a_zp", 0),
                weight_i8("W_b", vec![2, 2], &[5, 6, 7, 8]),
                scalar_f32("W_b_scale", 0.2),
                scalar_i8("W_b_zp", 0),
            ],
            ..GraphProto::default()
        };
        let mut m = fresh_model(graph);
        let r = apply(&mut m).unwrap();
        assert_eq!(r.dyn_quant_fusions_replaced, 2);
        let g = m.graph.unwrap();
        // Two MatMul + one Add survive; everything else is gone.
        let mm_count = g.node.iter().filter(|n| n.op_type == "MatMul").count();
        assert_eq!(mm_count, 2);
        assert!(r.is_clean());
    }
}