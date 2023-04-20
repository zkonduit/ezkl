use super::node::*;
use super::vars::*;
use super::GraphError;
use crate::circuit::ops::poly::PolyOp;
use crate::circuit::BaseConfig as PolyConfig;
use crate::circuit::Op;

use crate::commands::RunArgs;
use crate::commands::{Cli, Commands};
use crate::graph::scale_to_multiplier;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor};
use serde::Deserialize;
use serde::Serialize;
use tract_onnx::prelude::DatumExt;
use tract_onnx::prelude::Graph;
use tract_onnx::prelude::InferenceModelExt;
use tract_onnx::prelude::TypedFact;
use tract_onnx::prelude::TypedOp;
use tract_onnx::tract_hir::internal::Factoid;
use tract_onnx::tract_hir::internal::GenericFactoid;
//use clap::Parser;
use core::panic;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::error;
use log::{debug, info, trace};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;
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
    base: PolyConfig<F>,
    /// The model struct
    pub model: Model<F>,
    /// A wrapper for holding all columns that will be assigned to by the model
    pub vars: ModelVars<F>,
}

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug)]
pub struct Model<F: FieldExt + TensorType> {
    /// input indices
    pub inputs: Vec<usize>,
    /// output indices
    pub outputs: Vec<usize>,
    /// Graph of nodes we are loading from Onnx.
    pub nodes: NodeGraph<F>, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    /// The [RunArgs] being used
    pub run_args: RunArgs,
    /// The [Mode] we're using the model in.
    pub mode: Mode,
    /// Defines which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub visibility: VarVisibility,
}

impl<F: FieldExt + TensorType> Model<F> {
    /// Creates an `Model` from a specified path to an Onnx file.
    /// # Arguments
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
        let (model, nodes) = Self::load_onnx_model(path, run_args.scale, run_args.public_params)?;

        let om = Model {
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| o.node).collect(),
            run_args,
            nodes,
            mode,
            visibility,
        };

        Ok(om)
    }

    /// Runs a forward pass on sample data !
    /// # Arguments
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn forward(
        model_path: impl AsRef<Path>,
        model_inputs: &[Tensor<i128>],
        run_args: RunArgs,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        let (model, nodes) =
            Self::load_onnx_model(model_path, run_args.scale, run_args.public_params)?;

        let mut results: BTreeMap<&usize, Tensor<i128>> = BTreeMap::new();
        for (i, n) in nodes.iter() {
            let mut inputs = vec![];
            if n.opkind.is_input() {
                let mut t = model_inputs[*i].clone();
                t.reshape(&n.out_dims);
                inputs.push(t);
            } else {
                for i in n.inputs.iter() {
                    match results.get(&i) {
                        Some(value) => inputs.push(value.clone()),
                        None => return Err(Box::new(GraphError::MissingNode(*i))),
                    }
                }
            };
            results.insert(i, Op::<F>::f(&*n.opkind, &inputs)?);
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

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `path` - A path to an Onnx file.
    /// * `scale` - The scale to use for quantization.
    fn load_onnx_model(
        path: impl AsRef<Path>,
        scale: u32,
        public_params: bool,
    ) -> Result<(Graph<TypedFact, Box<dyn TypedOp>>, BTreeMap<usize, Node<F>>), Box<dyn Error>>
    {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|_| GraphError::ModelLoad)?;
        // .into_optimized()?;

        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node(id.node);

            // add batch dim
            let mut dims = vec![];
            let extracted_dims: Vec<usize> = input.outputs[0]
                .fact
                .shape
                .dims()
                .filter_map(|x| x.concretize())
                .map(|x| x.to_i64().unwrap() as usize)
                .collect();

            // if we have unknown / unspecified dims, add a batch dim of 1
            if let GenericFactoid::Only(elem) = input.outputs[0].fact.shape.rank() {
                if (elem as usize) > extracted_dims.len() {
                    dims.push(1);
                }
            };
            dims.extend(extracted_dims);

            model = model.with_input_fact(i, f32::fact(dims).into())?;
        }
        // Note: do not optimize the model, as the layout will depend on underlying hardware
        let model = model.into_typed()?.into_decluttered()?;

        println!("model {}", model);

        let mut nodes = BTreeMap::<usize, Node<F>>::new();
        for (i, n) in model.nodes.iter().enumerate() {
            let n = Node::<F>::new(n.clone(), &mut nodes, scale, public_params, i)?;
            nodes.insert(i, n);
        }

        debug!("\n {}", Table::new(nodes.iter()).to_string());

        Ok((model, nodes))
    }

    /// Creates a `Model` from parsed CLI arguments
    /// # Arguments
    /// * `cli` - [Cli]
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

    /// Creates a `Model` based on CLI arguments
    pub fn from_arg() -> Result<Self, Box<dyn Error>> {
        let conf = Cli::create()?;
        Self::from_ezkl_conf(conf)
    }

    /// Configures an `Model`. Does so one execution `bucket` at a time. Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// # Arguments
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the nodes loaded in `self.nodes`.
    pub fn configure(
        &self,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<ModelConfig<F>, Box<dyn Error>> {
        info!("configuring model");

        let mut base_gate = PolyConfig::configure(
            meta,
            vars.advices[0..2].try_into()?,
            &vars.advices[2],
            self.run_args.check_mode,
            self.run_args.tolerance as i32,
        );

        let lookup_ops: BTreeMap<&usize, &Node<F>> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.required_lookup().is_some())
            .collect();

        for node in lookup_ops.values() {
            self.conf_lookup(&mut base_gate, node, meta, vars)?;
        }

        Ok(ModelConfig {
            base: base_gate,
            model: self.clone(),
            vars: vars.clone(),
        })
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [Node] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_lookup(
        &self,
        config: &mut PolyConfig<F>,
        node: &Node<F>,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<(), Box<dyn Error>> {
        let input = &vars.advices[0];
        let output = &vars.advices[1];

        let op = match &node.opkind.required_lookup() {
            Some(nl) => nl.clone(),
            None => {
                return Err(Box::new(GraphError::WrongMethod(
                    node.idx,
                    node.opkind.as_str().to_string(),
                )));
            }
        };

        config.configure_lookup(meta, input, output, self.run_args.bits, &op)?;

        Ok(())
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    ///
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    pub fn layout(
        &self,
        mut config: ModelConfig<F>,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
        vars: &ModelVars<F>,
    ) -> Result<(), Box<dyn Error>> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for (i, input_idx) in self.inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(*input_idx, vars.instances[i].clone());
            } else {
                results.insert(*input_idx, inputs[i].clone());
            }
        }

        config.base.layout_tables(layouter)?;

        layouter.assign_region(
            || "model",
            |mut region| {
                let mut offset: usize = 0;
                for (idx, node) in self.nodes.iter() {
                    let values: Vec<ValTensor<F>> = node
                        .inputs
                        .iter()
                        .map(|i| results.get(i).unwrap().clone())
                        .collect_vec();

                    trace!("laying out offset {}", offset);
                    let res = config
                        .base
                        .layout(
                            Some(&mut region),
                            &values,
                            &mut offset,
                            node.opkind.clone_dyn(),
                        )
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?;

                    if let Some(vt) = res {
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
                    .map(|o| results.get(o).unwrap().clone())
                    .collect_vec();

                // pack outputs if need be
                if self.run_args.pack_base > 1 {
                    for i in 0..outputs.len() {
                        info!("packing outputs...");
                        outputs[i] = config
                            .base
                            .layout(
                                Some(&mut region),
                                &outputs[i..i + 1],
                                &mut offset,
                                Box::new(PolyOp::Pack(
                                    self.run_args.pack_base,
                                    self.run_args.scale,
                                )),
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
                }

                if self.run_args.public_outputs {
                    let _ = outputs
                        .into_iter()
                        .enumerate()
                        .map(|(i, output)| {
                            let mut instance_offset = 0;
                            if self.visibility.input.is_public() {
                                instance_offset += inputs.len();
                            };
                            config.base.layout(
                                Some(&mut region),
                                &[output, vars.instances[instance_offset + i].clone()],
                                &mut offset,
                                Box::new(PolyOp::RangeCheck(self.run_args.tolerance as i32)),
                            )
                        })
                        .collect_vec();
                }

                Ok(())
            },
        )?;
        info!("computing...");
        Ok(())
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    ///
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    pub fn dummy_layout(&self, input_shapes: &[Vec<usize>]) -> Result<usize, Box<dyn Error>> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();

        let inputs: Vec<ValTensor<F>> = input_shapes
            .iter()
            .map(|shape| {
                let t: Tensor<Value<F>> = Tensor::new(None, shape).unwrap();
                t.into()
            })
            .collect_vec();

        for (i, input_idx) in self.inputs.iter().enumerate() {
            results.insert(*input_idx, inputs[i].clone());
        }

        let mut dummy_config = PolyConfig::dummy(self.run_args.logrows as usize);

        let mut offset: usize = 0;
        for (idx, node) in self.nodes.iter() {
            let values: Vec<ValTensor<F>> = node
                .inputs
                .iter()
                .map(|i| results.get(i).unwrap().clone())
                .collect_vec();

            let res = dummy_config
                .layout(None, &values, &mut offset, node.opkind.clone_dyn())
                .map_err(|e| {
                    error!("{}", e);
                    halo2_proofs::plonk::Error::Synthesis
                })?;

            if let Some(vt) = res {
                // we get the max as for fused nodes this corresponds to the node output
                results.insert(*idx, vt);
            }
        }

        let output_nodes = self.outputs.iter();
        info!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let mut outputs = output_nodes
            .map(|o| results.get(o).unwrap().clone())
            .collect_vec();

        // pack outputs if need be
        if self.run_args.pack_base > 1 {
            for i in 0..outputs.len() {
                info!("packing outputs...");
                outputs[i] = dummy_config
                    .layout(
                        None,
                        &outputs[i..i + 1],
                        &mut offset,
                        Box::new(PolyOp::Pack(self.run_args.pack_base, self.run_args.scale)),
                    )
                    .map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?
                    .unwrap();
            }
        }

        if self.run_args.public_outputs {
            let _ = outputs
                .into_iter()
                .map(|output| {
                    dummy_config.layout(
                        None,
                        &[output.clone(), output],
                        &mut offset,
                        Box::new(PolyOp::RangeCheck(self.run_args.tolerance as i32)),
                    )
                })
                .collect_vec();
        }

        Ok(offset)
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
            .map(|o| self.nodes.get(o).unwrap().out_dims.clone())
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
            .map(|o| self.nodes.get(o).unwrap().out_dims.clone())
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<u32> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .map(|o| self.nodes.get(o).unwrap().out_scale)
            .collect_vec()
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
