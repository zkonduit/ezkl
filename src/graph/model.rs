use super::node::*;
use super::vars::*;
use super::GraphError;
use crate::circuit::BaseConfig as PolyConfig;
use crate::circuit::LookupOp;
use crate::circuit::Op as PolyOp;
use crate::circuit::OpKind;
use crate::commands::RunArgs;
use crate::commands::{Cli, Commands};
use crate::fieldutils::i128_to_felt;
use crate::graph::scale_to_multiplier;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor};
use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
//use clap::Parser;
use core::panic;
use halo2_proofs::circuit::Region;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::error;
use log::{debug, info, trace};
use std::cell::RefCell;
use std::cmp::max;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use tabled::Table;
use tract_onnx;
use tract_onnx::prelude::Framework;
/// Mode we're using the model in.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Mode {
    /// Initialize the model and display the operations table / graph
    Table,
    /// Initialize the model and generate a mock proof
    Mock,
    /// Initialize the model and generate a proof
    Prove,
    /// Initialize the model, generate a proof, and verify
    FullProve,
    /// Initialize the model and verify an already generated proof
    Verify,
}

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone, Debug)]
pub struct ModelConfig<F: FieldExt + TensorType> {
    configs: BTreeMap<usize, NodeConfig<F>>,
    /// The model struct
    pub model: Model,
    /// (optional) range checked outputs of the model graph
    pub range_checks: Vec<Rc<RefCell<PolyConfig<F>>>>,
    /// (optional) packed outputs of the model graph
    pub packed_outputs: Vec<Rc<RefCell<PolyConfig<F>>>>,
    /// A wrapper for holding all columns that will be assigned to by the model
    pub vars: ModelVars<F>,
}

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Model {
    /// input indices
    pub inputs: Vec<usize>,
    /// output indices
    pub outputs: Vec<usize>,
    /// Graph of nodes we are loading from Onnx.
    pub nodes: NodeGraph, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    /// The [RunArgs] being used
    pub run_args: RunArgs,
    /// The [Mode] we're using the model in.
    pub mode: Mode,
    /// Defines which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub visibility: VarVisibility,
}

impl Model {
    /// Creates an `Model` from a specified path to an Onnx file.
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    /// * `mode` - The [Mode] we're using the model in.
    /// * `visibility` - Which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub fn new(
        path: impl AsRef<Path>,
        run_args: RunArgs,
        mode: Mode,
        visibility: VarVisibility,
    ) -> Result<Self, Box<dyn Error>> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|_| GraphError::ModelLoad)?;
        info!("visibility: {}", visibility);

        let mut nodes = BTreeMap::<usize, Node>::new();
        for (i, n) in model.nodes.iter().enumerate() {
            let n = Node::new(n.clone(), &mut nodes, run_args.scale, i)?;
            nodes.insert(i, n);
        }
        let om = Model {
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| o.node).collect(),
            run_args,
            nodes,
            mode,
            visibility,
        };

        debug!("{}", Table::new(om.nodes.iter()).to_string());

        Ok(om)
    }

    /// Runs a dummy forward pass on sample data !
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn forward(
        model_path: impl AsRef<Path>,
        model_inputs: &[Tensor<i128>],
        run_args: RunArgs,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|_| GraphError::ModelLoad)?;
        info!("running forward pass");

        let mut nodes = BTreeMap::<usize, Node>::new();
        for (i, n) in model.nodes.iter().enumerate() {
            let n = Node::new(n.clone(), &mut nodes, run_args.scale, i)?;
            nodes.insert(i, n);
        }

        debug!("{}", Table::new(nodes.clone()).to_string());

        let mut results: BTreeMap<&usize, Tensor<i128>> = BTreeMap::new();
        for (i, n) in nodes.iter() {
            let mut inputs = vec![];
            for i in n.inputs.iter() {
                match results.get(&i) {
                    Some(value) => inputs.push(value.clone()),
                    None => return Err(Box::new(GraphError::MissingNode(*i))),
                }
            }
            match &n.opkind {
                OpKind::Lookup(op) => {
                    // assert_eq!(inputs.len(), 1);
                    results.insert(i, op.f(inputs[0].clone())?);
                }
                OpKind::Poly(op) => {
                    results.insert(i, op.f(inputs)?);
                }
                OpKind::Input => {
                    let mut t = model_inputs[*i].clone();
                    t.reshape(&n.out_dims);
                    results.insert(i, t);
                }
                OpKind::Const => {
                    results.insert(i, n.const_value.as_ref().unwrap().clone());
                }
                _ => {
                    panic!("unsupported op")
                }
            }
        }

        let output_nodes = model.outputs.iter();
        info!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().map(|o| o.node).collect_vec()
        );
        let outputs = output_nodes
            .map(|o| {
                let n = nodes.get(&o.node).unwrap();
                let scale = scale_to_multiplier(n.out_scale);
                results
                    .get(&o.node)
                    .unwrap()
                    .clone()
                    .map(|x| (x as f32) / scale)
            })
            .collect_vec();

        Ok(outputs)
    }

    /// Creates a `Model` from parsed CLI arguments
    pub fn from_ezkl_conf(cli: Cli) -> Result<Self, Box<dyn Error>> {
        let visibility = VarVisibility::from_args(cli.args.clone())?;
        match cli.command {
            Commands::Table { model } | Commands::Mock { model, .. } => {
                Model::new(model, cli.args, Mode::Mock, visibility)
            }
            Commands::Prove { model, .. }
            | Commands::Verify { model, .. }
            | Commands::Aggregate { model, .. } => {
                Model::new(model, cli.args, Mode::Prove, visibility)
            }
            #[cfg(not(target_arch = "wasm32"))]
            Commands::CreateEVMVerifier { model, .. } => {
                Model::new(model, cli.args, Mode::Prove, visibility)
            }
            #[cfg(feature = "render")]
            Commands::RenderCircuit { model, .. } => {
                Model::new(model, cli.args, Mode::Table, visibility)
            }
            _ => panic!(),
        }
    }

    ///
    pub fn write<W: Write>(&self, mut writer: BufWriter<W>) -> Result<(), Box<dyn Error>> {
        let circuit_bytes = bincode::serialize(&self)?;
        writer.write(&circuit_bytes)?;
        writer.flush()?;
        Ok(())
    }

    ///
    pub fn write_to_file(&self, path: PathBuf) -> Result<(), Box<dyn Error>> {
        let fs = File::create(path)?;
        let buffer = BufWriter::new(fs);
        self.write(buffer)
    }

    ///
    pub fn read<R: Read>(mut reader: BufReader<R>) -> Result<Self, Box<dyn Error>> {
        let buffer: &mut Vec<u8> = &mut vec![];
        reader.read_to_end(buffer)?;

        let circuit = bincode::deserialize(&buffer)?;
        Ok(circuit)
    }
    ///
    pub fn read_from_file(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        Self::read(reader)
    }

    /// Creates a `Model` based on CLI arguments
    pub fn from_arg() -> Result<Self, Box<dyn Error>> {
        let conf = Cli::create()?;
        Self::from_ezkl_conf(conf)
    }

    /// Configures an `Model`. Does so one execution `bucket` at a time. Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// # Arguments
    ///
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the nodes loaded in `self.nodes`.
    pub fn configure<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<ModelConfig<F>, Box<dyn Error>> {
        info!("configuring model");
        let mut results = BTreeMap::new();

        let mut base_gate = Rc::new(RefCell::new(PolyConfig::configure(
            meta,
            vars.advices[0..2].try_into()?,
            &vars.advices[2],
            self.run_args.check_mode,
            self.run_args.tolerance as i32,
        )));

        let non_op_nodes: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_const() || n.opkind.is_input())
            .collect();
        if !non_op_nodes.is_empty() {
            for (i, node) in non_op_nodes {
                let config = self.conf_non_op_node(node)?;
                results.insert(*i, config);
            }
        }

        // preserves ordering
        let poly_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_poly())
            .collect();
        // preserves ordering
        if !poly_ops.is_empty() {
            for (i, node) in poly_ops {
                let config = self.conf_poly_ops(node, &mut base_gate)?;
                results.insert(*i, config);

                let mut display: String = "Poly nodes: ".to_string();
                display.push_str(&format!("| {} ({:?}) | ", i, node.opkind));

                trace!("{}", display);
            }
        }

        let lookup_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_lookup())
            .collect();

        if !lookup_ops.is_empty() {
            for (i, node) in lookup_ops {
                let config = self.conf_lookup(base_gate.clone(), node, meta, vars)?;
                results.insert(*i, config);
            }
        }

        let mut range_checks = vec![];
        let mut packed_outputs = vec![];
        if self.run_args.pack_base > 1 {
            info!("packing outputs...");
            packed_outputs = self.output_ops(&mut base_gate);
        }
        if self.visibility.output.is_public() {
            range_checks = self.output_ops(&mut base_gate);
        };

        Ok(ModelConfig {
            configs: results,
            model: self.clone(),
            range_checks,
            packed_outputs,
            vars: vars.clone(),
        })
    }

    fn output_ops<F: FieldExt + TensorType>(
        &self,
        base_gate: &mut Rc<RefCell<PolyConfig<F>>>,
    ) -> Vec<Rc<RefCell<PolyConfig<F>>>> {
        let mut configs = vec![];

        for _ in self.output_shapes() {
            configs.push(base_gate.clone());
        }

        configs
    }

    /// Configures non op related nodes (eg. representing an input or const value)
    pub fn conf_non_op_node<F: FieldExt + TensorType>(
        &self,
        node: &Node,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        match &node.opkind {
            OpKind::Const => {
                // Typically parameters for one or more layers.
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(NodeConfig::Const)
            }
            OpKind::Input => {
                // This is the input to the model (e.g. the image).
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(NodeConfig::Input)
            }
            OpKind::Unknown(_c) => {
                unimplemented!()
            }
            c => Err(Box::new(GraphError::WrongMethod(node.idx, c.clone()))),
        }
    }

    /// Configures a [BTreeMap] of operations that can be constrained using polynomials. These correspond to operations that are represented in
    /// the `circuit::polynomial` module. A single configuration is output, representing the amalgamation of these operations into
    /// a single Halo2 gate.
    /// # Arguments
    ///
    /// * `nodes` - A [BTreeMap] of (node index, [Node] pairs). The [Node] must represent a polynomial op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_poly_ops<F: FieldExt + TensorType>(
        &self,
        node: &Node,
        base_gate: &mut Rc<RefCell<PolyConfig<F>>>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        let input_nodes = node
            .inputs
            .iter()
            .map(|i| self.nodes.get(&i).unwrap())
            .collect_vec();

        let input_idx = input_nodes.iter().map(|f| f.idx).collect_vec();

        let config = NodeConfig::Op {
            config: base_gate.clone(),
            inputs: input_idx,
            op: node.opkind.clone(),
        };
        Ok(config)
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [Node] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_lookup<F: FieldExt + TensorType>(
        &self,
        config: Rc<RefCell<PolyConfig<F>>>,
        node: &Node,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        let input = &vars.advices[0];
        let output = &vars.advices[1];
        let input_nodes = node
            .inputs
            .iter()
            .map(|i| self.nodes.get(&i).unwrap())
            .collect_vec();

        let input_idx = input_nodes.iter().map(|f| f.idx).collect_vec();

        let mut op = match &node.opkind {
            OpKind::Lookup(l) => l.clone(),
            c => {
                return Err(Box::new(GraphError::WrongMethod(node.idx, c.clone())));
            }
        };

        match op {
            LookupOp::PReLU { scale, .. } => {
                op = LookupOp::ReLU { scale };
            }
            LookupOp::Max | LookupOp::Min | LookupOp::MaxPool2d { .. } => {
                op = LookupOp::ReLU { scale: 1 };
            }
            LookupOp::Mean { scale } => {
                assert_eq!(input_nodes.len(), 1);
                op = LookupOp::Div {
                    denom: crate::circuit::utils::F32(
                        // we need to scale the denom by the number of elements in the input tensor and the calculated scale diff
                        (scale * input_nodes[0].out_dims.iter().product::<usize>()) as f32,
                    ),
                };
            }
            _ => {}
        }

        config
            .borrow_mut()
            .configure_lookup(meta, input, output, self.run_args.bits, &op)?;

        let config = NodeConfig::Op {
            config,
            inputs: input_idx,
            op: node.opkind.clone(),
        };
        Ok(config)
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    ///
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    pub fn layout<F: FieldExt + TensorType>(
        &self,
        mut config: ModelConfig<F>,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
        vars: &ModelVars<F>,
    ) -> Result<(), Box<dyn Error>> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for (i, input_value) in inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(i, vars.instances[i].clone());
            } else {
                results.insert(i, input_value.clone());
            }
        }

        // layout any lookup tables
        let _: Vec<()> = config
            .configs
            .values()
            .map(|c| match c {
                // only lays out tables if they exist so this can be called safely
                NodeConfig::Op { config, .. } => config.borrow_mut().layout_tables(layouter),
                _ => Ok(()),
            })
            .collect::<Result<Vec<()>, _>>()?;

        layouter.assign_region(
            || "model",
            |mut region| {
                let mut offset: usize = 0;
                for (idx, config) in config.configs.iter() {
                    trace!("laying out offset {}", offset);
                    if let Some(vt) = self
                        .layout_config(&mut region, &mut results, config, &mut offset)
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?
                    {
                        // we get the max as for fused nodes this corresponds to the node output
                        results.insert(*idx, vt);
                        //only use with mock prover
                        if matches!(self.mode, Mode::Mock) {
                            trace!(
                                "------------ output node {:?}: {:?}",
                                idx,
                                results.get(idx).unwrap().show()
                            );
                        }
                    }
                }

                let output_nodes = self.outputs.iter();
                info!(
                    "model outputs are nodes: {:?}",
                    output_nodes.clone().collect_vec()
                );
                let mut outputs = output_nodes
                    .map(|o| results.get(&o).unwrap().clone())
                    .collect_vec();

                // pack outputs if need be
                for (i, packed_output) in config.packed_outputs.iter_mut().enumerate() {
                    info!("packing outputs...");
                    outputs[i] = packed_output
                        .borrow_mut()
                        .layout(
                            &mut region,
                            &outputs[i..i + 1],
                            &mut offset,
                            PolyOp::Pack(self.run_args.pack_base, self.run_args.scale).into(),
                        )
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?
                        .unwrap();
                    // only use with mock prover
                    if matches!(self.mode, Mode::Mock) {
                        trace!("------------ packed output {:?}", outputs[i].show());
                    }
                }

                let _ = config
                    .range_checks
                    .iter()
                    .zip(outputs)
                    .enumerate()
                    .map(|(i, (range_check, output))| {
                        let mut instance_offset = 0;
                        if self.visibility.input.is_public() {
                            instance_offset += inputs.len();
                        };
                        range_check.borrow_mut().layout(
                            &mut region,
                            &[output, vars.instances[instance_offset + i].clone()],
                            &mut offset,
                            PolyOp::RangeCheck(self.run_args.tolerance as i32).into(),
                        )
                    })
                    .collect_vec();
                Ok(())
            },
        )?;
        info!("computing...");
        Ok(())
    }

    /// Assigns values to a single region, represented as a [NodeConfig].
    /// # Arguments
    ///
    /// * `config` - [NodeConfig] the single region we will layout.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - [BTreeMap] of values to feed into the [NodeConfig], can also include previous intermediate results, i.e the output of other nodes.
    fn layout_config<F: FieldExt + TensorType>(
        &self,
        region: &mut Region<F>,
        inputs: &mut BTreeMap<usize, ValTensor<F>>,
        config: &NodeConfig<F>,
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        // The node kind and the config should be the same.
        let res = match config.clone() {
            NodeConfig::Op {
                config,
                inputs: idx,
                op,
            } => {
                let values: Vec<ValTensor<F>> = idx
                    .iter()
                    .map(|i| {
                        let node = &self.nodes.get(i).unwrap();
                        match node.opkind {
                            OpKind::Const => {
                                let val = node
                                    .const_value
                                    .clone()
                                    .context("Tensor<i128> should already be loaded")
                                    .unwrap();
                                if self.visibility.params.is_public() {
                                    val.map(|x| {
                                        crate::tensor::ValType::Constant(i128_to_felt::<F>(x))
                                    })
                                    .into()
                                } else {
                                    val.map(|x| {
                                        crate::tensor::ValType::Value(Value::known(
                                            i128_to_felt::<F>(x),
                                        ))
                                    })
                                    .into()
                                }
                            }
                            _ => inputs.get(i).unwrap().clone(),
                        }
                    })
                    .collect_vec();

                let res = config.borrow_mut().layout(region, &values, offset, op)?;

                res
            }
            NodeConfig::Input => None,
            NodeConfig::Const => None,
            _ => {
                return Err(Box::new(GraphError::UnsupportedOp));
            }
        };
        Ok(res)
    }

    /// Returns the number of the computational graph's inputs
    pub fn num_inputs(&self) -> usize {
        let input_nodes = self.inputs.iter();
        input_nodes.len()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .map(|o| self.nodes.get(&o).unwrap().out_dims.clone())
            .collect_vec()
    }

    /// Returns the number of the computational graph's outputs
    pub fn num_outputs(&self) -> usize {
        let output_nodes = self.outputs.iter();
        output_nodes.len()
    }

    /// Returns shapes of the computational graph's outputs
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.outputs
            .iter()
            .map(|o| self.nodes.get(&o).unwrap().out_dims.clone())
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<u32> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .map(|o| self.nodes.get(&o).unwrap().out_scale)
            .collect_vec()
    }

    /// Total number of variables in lookup layers
    pub fn num_vars_lookup_op(&self, lookup_op: &LookupOp) -> Vec<usize> {
        let mut count = BTreeMap::<LookupOp, (usize, usize)>::new();
        for (_, n) in self.nodes.iter() {
            if n.opkind == OpKind::Lookup(lookup_op.clone()) {
                match &n.opkind {
                    OpKind::Lookup(op) => {
                        let elem = count.get_mut(op);
                        // handle output variables
                        let output_size: usize = n.out_dims.iter().product();
                        let input_size = output_size;
                        match elem {
                            None => {
                                count.insert(op.clone(), (input_size, output_size));
                            }
                            Some(m) => {
                                m.0 += input_size;
                                m.1 += output_size;
                            }
                        }
                    }
                    // should never reach here
                    _ => panic!(),
                }
            }
        }
        // now get the max across all ops
        let (mut num_inputs, mut num_outputs) = (0, 0);
        for (_, v) in count.iter() {
            num_inputs = max(num_inputs, v.0);
            num_outputs = max(num_outputs, v.1);
        }
        vec![num_inputs, num_outputs]
    }

    /// Maximum number of input variables
    pub fn total_var_len(&self) -> usize {
        let mut maximum_var_len = 0;

        let poly_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_poly())
            .collect();

        let _: Vec<_> = poly_ops
            .values()
            .map(|n| match &n.opkind {
                OpKind::Poly(p) => {
                    let in_dims = n
                        .inputs
                        .iter()
                        .map(|i| self.nodes.get(&i).unwrap().out_dims.clone());
                    let layout_shape = p.circuit_shapes(in_dims.collect_vec());
                    maximum_var_len += layout_shape.last().unwrap();
                }
                _ => panic!(),
            })
            .collect();

        let lookup_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_lookup())
            .collect();

        for op in lookup_ops {
            let len = (*op.1.out_dims).iter().product::<usize>();
            maximum_var_len += len;
        }

        let output_lens: usize = self
            .output_shapes()
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>();

        let input_lens: usize = self
            .input_shapes()
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>();

        if self.run_args.pack_base > 1 {
            maximum_var_len += output_lens;
        }
        if matches!(self.visibility.output, Visibility::Public) {
            maximum_var_len += output_lens;
        }
        if matches!(self.visibility.output, Visibility::Public) {
            maximum_var_len += input_lens;
        }

        maximum_var_len
    }

    /// Number of instances used by the circuit
    pub fn instance_shapes(&self) -> Vec<Vec<usize>> {
        // for now the number of instances corresponds to the number of graph / model outputs
        let mut instance_shapes = vec![];
        if self.visibility.input.is_public() {
            instance_shapes.extend(self.input_shapes());
        }
        if self.visibility.output.is_public() {
            instance_shapes.extend(self.output_shapes());
        }
        instance_shapes
    }
}
