use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::fieldutils::i128_to_felt;
use crate::tensor::{Tensor, ValTensor};
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{ConstraintSystem, Error};
use halo2curves::bn256::Fr as Fp;
use serde::{Deserialize, Serialize};

use super::GraphInput;
use super::{VarVisibility, Visibility};

const POSEIDON_LEN_GRAPH: usize = 10;

// TODO: make this a function of the number of constraints this is a bit of a hack
const POSEIDON_CONSTRAINTS_ESTIMATE: usize = 44;

/// Poseidon module type
pub type ModulePoseidon =
    PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>;
/// Poseidon module config
pub type ModulePoseidonConfig = PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>;

///
#[derive(Clone, Debug, Default)]
pub struct ModuleConfigs {
    /// Poseidon
    poseidon: Option<ModulePoseidonConfig>,
}

impl ModuleConfigs {
    /// Create new module configs from visibility of each variable
    pub fn from_visibility(
        cs: &mut ConstraintSystem<Fp>,
        visibility: VarVisibility,
        num_module_instances: usize,
    ) -> Self {
        let poseidon_config = if (visibility.input.is_hashed()
            || visibility.output.is_hashed()
            || visibility.params.is_hashed())
            && num_module_instances > 0
        {
            Some(ModulePoseidon::configure(cs))
        } else {
            None
        };
        Self {
            poseidon: poseidon_config,
        }
    }
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleForwardResult {
    /// The inputs of the forward pass
    pub poseidon_hash: Vec<Fp>,
}

/// Graph modules that can process inputs, params and outputs beyond the basic operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphModules;

impl GraphModules {
    /// Get the instances for the module
    fn instances_from_visibility(
        visibility: Visibility,
        module_res: &Option<ModuleForwardResult>,
    ) -> Vec<Fp> {
        if visibility.is_hashed() {
            module_res.clone().unwrap().poseidon_hash
        } else {
            vec![]
        }
    }

    /// Generate the public inputs for the circuit
    pub fn public_inputs(data: &GraphInput, visibility: VarVisibility) -> Vec<Fp> {
        let mut pi = vec![];
        pi.extend(Self::instances_from_visibility(
            visibility.input,
            &data.processed_inputs,
        ));
        pi.extend(Self::instances_from_visibility(
            visibility.params,
            &data.processed_params,
        ));
        pi.extend(Self::instances_from_visibility(
            visibility.output,
            &data.processed_outputs,
        ));
        pi
    }

    fn num_constraint_given_shapes(
        visibility: Visibility,
        shapes: Vec<Vec<usize>>,
    ) -> (usize, usize) {
        let mut num_constraints = 0;
        let mut num_instances = 0;
        if visibility.is_hashed() {
            for shape in shapes {
                let total_len = shape.iter().product::<usize>();
                num_constraints += POSEIDON_CONSTRAINTS_ESTIMATE * total_len;
                if total_len > 0 {
                    num_instances += 1;
                }
            }
        }
        (num_constraints, num_instances)
    }
    /// Get the number of constraints and instances for the module
    pub fn num_constraints_and_instances(
        input_shapes: Vec<Vec<usize>>,
        params_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        visibility: VarVisibility,
    ) -> (usize, usize) {
        let (num_constraints_input, num_instances_input) =
            Self::num_constraint_given_shapes(visibility.input, input_shapes);

        let (num_constraints_params, num_instances_params) =
            Self::num_constraint_given_shapes(visibility.params, params_shapes);

        let (num_constraints_output, num_instances_output) =
            Self::num_constraint_given_shapes(visibility.output, output_shapes);

        let num_constraints =
            num_constraints_input + num_constraints_params + num_constraints_output;
        let num_instances = num_instances_input + num_instances_params + num_instances_output;

        (num_constraints, num_instances)
    }

    /// Layout the module
    fn layout_module(
        module: &impl Module<Fp>,
        layouter: &mut impl Layouter<Fp>,
        values: &mut [ValTensor<Fp>],
        instance_offset: &mut usize,
    ) -> Result<(), Error> {
        // reserve module 0 for ... modules
        layouter.assign_region(|| "_enter_module_0", |_| Ok(()))?;

        values.iter_mut().enumerate().for_each(|(i, x)| {
            // hash the input and replace the constrained cells in the input
            let cloned_x = (*x).clone();
            let dims = cloned_x.dims();
            *x = module.layout(layouter, x, *instance_offset + i).unwrap();
            x.reshape(dims).unwrap();
            // increment the instance offset to make way for future module layouts
            *instance_offset += module.num_instances();
        });
        Ok(())
    }

    /// Layout the module
    pub fn layout(
        layouter: &mut impl Layouter<Fp>,
        configs: &ModuleConfigs,
        values: &mut [ValTensor<Fp>],
        element_visibility: Visibility,
        instance_offset: &mut usize,
    ) -> Result<(), Error> {
        if element_visibility.is_hashed() && values.len() > 0 {
            let poseidon_config = configs.poseidon.clone().unwrap();
            let chip = ModulePoseidon::new(poseidon_config);
            Self::layout_module(&chip, layouter, values, instance_offset)
        } else {
            Ok(())
        }
    }

    /// Run forward pass
    pub fn forward(
        inputs: &[Tensor<i128>],
    ) -> Result<ModuleForwardResult, Box<dyn std::error::Error>> {
        let mut outputs = vec![];
        for input in inputs.iter() {
            if input.len() > 0 {
                outputs.push(
                    ModulePoseidon::run(input.iter().map(|x| i128_to_felt(*x)).collect())?[0],
                );
            }
        }
        Ok(ModuleForwardResult {
            poseidon_hash: outputs,
        })
    }
}
