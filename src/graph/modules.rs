use crate::circuit::modules::elgamal::{ElGamalConfig, ElGamalGadget, ElGamalVariables};
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::tensor::{Tensor, ValTensor, ValType};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{ConstraintSystem, Error};
use halo2curves::bn256::Fr as Fp;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::GraphWitness;
use super::{VarVisibility, Visibility};

/// poseidon len to hash in tree
pub const POSEIDON_LEN_GRAPH: usize = 32;

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
    /// ElGamal
    elgamal: Option<ElGamalConfig>,
}

impl ModuleConfigs {
    /// Create new module configs from visibility of each variable
    pub fn from_visibility(
        cs: &mut ConstraintSystem<Fp>,
        visibility: VarVisibility,
        module_size: ModuleSizes,
    ) -> Self {
        let mut config = Self::default();

        if (visibility.input.is_hashed()
            || visibility.output.is_hashed()
            || visibility.params.is_hashed())
            && module_size.poseidon.1[0] > 0
        {
            config.poseidon = Some(ModulePoseidon::configure(cs))
        };

        if (visibility.input.is_encrypted()
            || visibility.output.is_encrypted()
            || visibility.params.is_encrypted())
            && module_size.elgamal.1[0] > 0
        {
            config.elgamal = Some(ElGamalGadget::configure(cs))
        };
        config
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Module variable settings
pub struct ModuleVarSettings {
    ///
    elgamal: Option<ElGamalVariables>,
}

impl ModuleVarSettings {
    /// Create new module variable settings
    pub fn new(elgamal: ElGamalVariables) -> Self {
        ModuleVarSettings {
            elgamal: Some(elgamal),
        }
    }
}

impl Default for ModuleVarSettings {
    fn default() -> Self {
        let dummy_elgamal = ElGamalVariables::default();
        ModuleVarSettings {
            elgamal: Some(dummy_elgamal),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// Module input settings
pub struct ModuleSettings {
    ///
    pub input: ModuleVarSettings,
    ///
    pub params: ModuleVarSettings,
    ///
    pub output: ModuleVarSettings,
}

impl From<&GraphWitness> for ModuleSettings {
    fn from(graph_input: &GraphWitness) -> Self {
        let mut settings = Self::default();

        if let Some(processed_inputs) = &graph_input.processed_inputs {
            if let Some(elgamal_result) = &processed_inputs.elgamal {
                settings.input = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }
        if let Some(processed_params) = &graph_input.processed_params {
            if let Some(elgamal_result) = &processed_params.elgamal {
                settings.params = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }
        if let Some(processed_outputs) = &graph_input.processed_outputs {
            if let Some(elgamal_result) = &processed_outputs.elgamal {
                settings.output = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }

        settings
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// Result from ElGamal
pub struct ElGamalResult {
    /// ElGamal variables
    pub variables: ElGamalVariables,
    /// ElGamal ciphertexts
    pub ciphertexts: Vec<Vec<Fp>>,
    /// ElGamal encrypted message
    pub encrypted_messages: Vec<Vec<Fp>>,
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleForwardResult {
    /// The inputs of the forward pass for poseidon
    pub poseidon_hash: Option<Vec<Fp>>,
    /// The outputs of the forward pass for ElGamal
    pub elgamal: Option<ElGamalResult>,
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleInstances {
    poseidon: Vec<Fp>,
    elgamal: Vec<Fp>,
}

impl ModuleInstances {
    /// Flatten the instances in order of module config
    pub fn flatten(&self) -> Vec<Vec<Fp>> {
        let mut instances = vec![];
        // check if poseidon is empty
        if !self.poseidon.is_empty() {
            // we push as its a 1D vector
            instances.push(self.poseidon.clone());
        }
        if !self.elgamal.is_empty() {
            // we extend as its a 2D vector
            instances.push(self.elgamal.clone());
        }
        instances
    }
}

/// Offset for the instances
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleInstanceOffset {
    poseidon: Vec<usize>,
    elgamal: Vec<usize>,
}

impl ModuleInstanceOffset {
    ///
    pub fn new() -> Self {
        ModuleInstanceOffset {
            poseidon: vec![0; crate::circuit::modules::poseidon::NUM_INSTANCE_COLUMNS],
            elgamal: vec![0; crate::circuit::modules::elgamal::NUM_INSTANCE_COLUMNS],
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
///
pub struct ModuleSizes {
    poseidon: (usize, Vec<usize>),
    elgamal: (usize, Vec<usize>),
}

impl ModuleSizes {
    /// Create new module sizes
    pub fn new() -> Self {
        ModuleSizes {
            poseidon: (
                0,
                vec![0; crate::circuit::modules::poseidon::NUM_INSTANCE_COLUMNS],
            ),
            elgamal: (
                0,
                vec![0; crate::circuit::modules::elgamal::NUM_INSTANCE_COLUMNS],
            ),
        }
    }

    /// Get the number of constraints
    pub fn max_constraints(&self) -> usize {
        self.poseidon.0.max(self.elgamal.0)
    }
    /// Get the number of instances
    pub fn num_instances(&self) -> Vec<usize> {
        // concat
        self.poseidon
            .1
            .iter()
            .chain(self.elgamal.1.iter())
            .copied()
            .collect_vec()
    }
}

/// Graph modules that can process inputs, params and outputs beyond the basic operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphModules;

impl GraphModules {
    /// Get the instances for the module
    fn instances_from_visibility(
        visibility: Visibility,
        module_res: &Option<ModuleForwardResult>,
        instances: &mut ModuleInstances,
    ) {
        if let Some(res) = module_res {
            if visibility.is_hashed() {
                instances
                    .poseidon
                    .extend(res.poseidon_hash.clone().unwrap());
            } else if visibility.is_encrypted() {
                instances.elgamal.extend(
                    res.elgamal
                        .clone()
                        .unwrap()
                        .ciphertexts
                        .into_iter()
                        .flatten(),
                );
            }
        }
    }

    /// Generate the public inputs for the circuit
    pub fn public_inputs(data: &GraphWitness, visibility: VarVisibility) -> Vec<Vec<Fp>> {
        let mut instances = ModuleInstances::default();
        Self::instances_from_visibility(visibility.input, &data.processed_inputs, &mut instances);
        Self::instances_from_visibility(visibility.params, &data.processed_params, &mut instances);
        Self::instances_from_visibility(visibility.output, &data.processed_outputs, &mut instances);

        instances.flatten()
    }

    fn num_constraint_given_shapes(
        visibility: Visibility,
        shapes: Vec<Vec<usize>>,
        sizes: &mut ModuleSizes,
    ) {
        for shape in shapes {
            let total_len = shape.iter().product::<usize>();
            if total_len > 0 {
                if visibility.is_hashed() {
                    sizes.poseidon.0 += ModulePoseidon::num_rows(total_len);
                    // 1 constraints for hash
                    sizes.poseidon.1[0] += 1;
                } else if visibility.is_encrypted() {
                    // add the 1 time fixed cost of maingate + ecc chips
                    sizes.elgamal.0 += ElGamalGadget::num_rows(total_len);
                    // 4 constraints for each ciphertext c1, c2, and sk
                    sizes.elgamal.1[0] += 4;
                }
            }
        }
    }
    /// Get the number of constraints and instances for the module
    pub fn num_constraints_and_instances(
        input_shapes: Vec<Vec<usize>>,
        params_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        visibility: VarVisibility,
    ) -> ModuleSizes {
        let mut module_sizes = ModuleSizes::new();

        Self::num_constraint_given_shapes(visibility.input, input_shapes, &mut module_sizes);
        Self::num_constraint_given_shapes(visibility.params, params_shapes, &mut module_sizes);
        Self::num_constraint_given_shapes(visibility.output, output_shapes, &mut module_sizes);

        module_sizes
    }

    /// Layout the module
    fn layout_module(
        module: &impl Module<Fp>,
        layouter: &mut impl Layouter<Fp>,
        values: &mut [Vec<ValTensor<Fp>>],
        instance_offset: &mut [usize],
    ) -> Result<(), Error> {
        // reserve module 0 for ... modules
        values.iter_mut().for_each(|x| {
            // hash the input and replace the constrained cells in the input
            let cloned_x = (*x).clone();
            let dims = cloned_x[0].dims();
            x[0] = module
                .layout(layouter, &cloned_x, instance_offset.to_owned())
                .unwrap();
            x[0].reshape(dims).unwrap();
            for (i, inc) in module.instance_increment_input().iter().enumerate() {
                // increment the instance offset to make way for future module layouts
                instance_offset[i] += inc;
            }
        });

        Ok(())
    }

    /// Layout the module
    pub fn layout(
        layouter: &mut impl Layouter<Fp>,
        configs: &ModuleConfigs,
        values: &mut [ValTensor<Fp>],
        element_visibility: Visibility,
        instance_offset: &mut ModuleInstanceOffset,
        module_settings: &ModuleVarSettings,
    ) -> Result<(), Error> {
        // If the module is hashed, then we need to hash the inputs
        if element_visibility.is_hashed() && !values.is_empty() {
            // reserve module 0 for poseidon modules
            layouter.assign_region(|| "_enter_module_0", |_| Ok(()))?;
            // config for poseidon
            let poseidon_config = configs.poseidon.clone().unwrap();
            // create the module
            let chip = ModulePoseidon::new(poseidon_config);
            // concat values and sk to get the inputs
            let mut inputs = values.iter_mut().map(|x| vec![x.clone()]).collect_vec();
            // layout the module
            Self::layout_module(&chip, layouter, &mut inputs, &mut instance_offset.poseidon)?;
            // replace the inputs with the outputs
            values.iter_mut().enumerate().for_each(|(i, x)| {
                x.clone_from(&inputs[i][0]);
            });

        // If the module is encrypted, then we need to encrypt the inputs
        } else if element_visibility.is_encrypted() && !values.is_empty() {
            // reserve module 1 for elgamal modules
            layouter.assign_region(|| "_enter_module_1", |_| Ok(()))?;
            // config for elgamal
            let elgamal_config = configs.elgamal.clone().unwrap();
            // create the module
            let mut chip = ElGamalGadget::new(elgamal_config);
            // load the variables
            let variables = module_settings.elgamal.as_ref().unwrap().clone();
            chip.load_variables(variables.clone());
            // load the sk:
            let sk: Tensor<ValType<Fp>> =
                Tensor::new(Some(&[Value::known(variables.sk).into()]), &[1]).unwrap();
            // concat values and sk to get the inputs
            let mut inputs = values
                .iter_mut()
                .map(|x| vec![x.clone(), sk.clone().into()])
                .collect_vec();
            // layout the module
            Self::layout_module(&chip, layouter, &mut inputs, &mut instance_offset.elgamal)?;
            // replace the inputs with the outputs
            values.iter_mut().enumerate().for_each(|(i, x)| {
                x.clone_from(&inputs[i][0]);
            });
        }

        Ok(())
    }

    /// Run forward pass
    pub fn forward(
        inputs: &[Tensor<Fp>],
        element_visibility: Visibility,
    ) -> Result<ModuleForwardResult, Box<dyn std::error::Error>> {
        let mut rng = &mut rand::thread_rng();
        let mut poseidon_hash = None;
        let mut elgamal = None;

        if element_visibility.is_hashed() {
            let field_elements = inputs.iter().fold(vec![], |mut acc, x| {
                let res = ModulePoseidon::run(x.to_vec()).unwrap()[0].clone();
                acc.extend(res);
                acc
            });
            poseidon_hash = Some(field_elements);
        }

        if element_visibility.is_encrypted() {
            let variables = ElGamalVariables::gen_random(&mut rng);
            let ciphertexts = inputs.iter().fold(vec![], |mut acc, x| {
                let res = ElGamalGadget::run((x.to_vec(), variables.clone())).unwrap();
                acc.extend(res);
                acc
            });

            let encrypted_messages = inputs.iter().fold(vec![], |mut acc, x| {
                let res = ElGamalGadget::encrypt(variables.pk, x.to_vec(), variables.r).c2;
                acc.push(res);
                acc
            });

            elgamal = Some(ElGamalResult {
                variables,
                ciphertexts,
                encrypted_messages,
            });
        }

        Ok(ModuleForwardResult {
            poseidon_hash,
            elgamal,
        })
    }
}
