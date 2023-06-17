use crate::circuit::modules::elgamal::{ElGamalConfig, ElGamalGadget, ElGamalVariables};
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::fieldutils::i128_to_felt;
use crate::tensor::{Tensor, ValTensor, ValType};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{ConstraintSystem, Error};
use halo2curves::bn256::Fr as Fp;
use itertools::Itertools;
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
            && module_size.elgamal.1[2] > 0
        {
            config.elgamal = Some(ElGamalGadget::configure(cs))
        };
        config
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

impl From<&GraphInput> for ModuleSettings {
    fn from(graph_input: &GraphInput) -> Self {
        let mut settings = Self::default();

        if let Some(processed_inputs) = &graph_input.processed_inputs {
            if let Some(elgamal_result) = &processed_inputs.elgmal_results {
                settings.input = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }
        if let Some(processed_params) = &graph_input.processed_params {
            if let Some(elgamal_result) = &processed_params.elgmal_results {
                settings.params = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }
        if let Some(processed_outputs) = &graph_input.processed_outputs {
            if let Some(elgamal_result) = &processed_outputs.elgmal_results {
                settings.output = ModuleVarSettings::new(elgamal_result.variables.clone());
            }
        }

        settings
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// Result from ElGamal
pub struct ElGamalResult {
    variables: ElGamalVariables,
    ciphertexts: Vec<Vec<Fp>>,
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleForwardResult {
    /// The inputs of the forward pass for poseidon
    pub poseidon_hash: Option<Vec<Fp>>,
    /// The outputs of the forward pass for ElGamal
    pub elgmal_results: Option<ElGamalResult>,
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleInstances {
    poseidon: Vec<Fp>,
    elgamal: Vec<Vec<Fp>>,
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
            instances.extend(self.elgamal.clone());
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
            .map(|x| *x)
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
        if visibility.is_hashed() {
            instances
                .poseidon
                .extend(module_res.clone().unwrap().poseidon_hash.unwrap());
        } else if visibility.is_encrypted() {
            let ciphers = module_res
                .clone()
                .unwrap()
                .elgmal_results
                .unwrap()
                .ciphertexts;
            if instances.elgamal.is_empty() {
                instances.elgamal = ciphers;
            } else {
                if !ciphers[2].is_empty() {
                    for i in 0..instances.elgamal.len() {
                        instances.elgamal[i].extend(ciphers[i].clone());
                    }
                }
            }
        }
    }

    /// Generate the public inputs for the circuit
    pub fn public_inputs(data: &GraphInput, visibility: VarVisibility) -> Vec<Vec<Fp>> {
        let mut instances = ModuleInstances::default();
        Self::instances_from_visibility(visibility.input, &data.processed_inputs, &mut instances);
        Self::instances_from_visibility(visibility.params, &data.processed_params, &mut instances);
        Self::instances_from_visibility(visibility.output, &data.processed_outputs, &mut instances);

        println!("instances: {:?}", instances.flatten());

        instances.flatten()
    }

    fn num_constraint_given_shapes(
        visibility: Visibility,
        shapes: Vec<Vec<usize>>,
        sizes: &mut ModuleSizes,
    ) {
        if visibility.is_hashed() {
            for shape in shapes {
                let total_len = shape.iter().product::<usize>();
                sizes.poseidon.0 += POSEIDON_CONSTRAINTS_ESTIMATE * total_len;
                if total_len > 0 {
                    sizes.poseidon.1[0] += 1;
                }
            }
        } else if visibility.is_encrypted() {
            let total_len = shapes
                .iter()
                .map(|x| x.iter().product::<usize>())
                .sum::<usize>();
            if total_len > 0 {
                sizes.elgamal.1[1] += 1;
                sizes.elgamal.1[3] += 1;
            }
            for shape in shapes {
                let total_len = shape.iter().product::<usize>();
                sizes.elgamal.0 += POSEIDON_CONSTRAINTS_ESTIMATE * total_len;
                if total_len > 0 {
                    sizes.elgamal.1[2] += 1;
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
        module: &mut impl Module<Fp>,
        layouter: &mut impl Layouter<Fp>,
        values: &mut [Vec<ValTensor<Fp>>],
        instance_offset: &mut Vec<usize>,
    ) -> Result<(), Error> {
        // reserve module 0 for ... modules

        values.iter_mut().for_each(|x| {
            // hash the input and replace the constrained cells in the input
            let cloned_x = (*x).clone();
            let dims = cloned_x[0].dims();
            x[0] = module
                .layout(layouter, &cloned_x, instance_offset.clone())
                .unwrap();
            x[0].reshape(dims).unwrap();
            for (i, inc) in module
                .instance_increment_input(x.iter().map(|x| x.len()).collect())
                .iter()
                .enumerate()
            {
                instance_offset[i] += inc;
            }
            // increment the instance offset to make way for future module layouts
        });

        for (i, inc) in module.instance_increment_module().iter().enumerate() {
            instance_offset[i] += inc;
        }

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
        if element_visibility.is_hashed() && values.len() > 0 {
            // reserve module 0 for poseidon modules
            layouter.assign_region(|| "_enter_module_0", |_| Ok(()))?;
            // config for poseidon
            let poseidon_config = configs.poseidon.clone().unwrap();
            // create the module
            let mut chip = ModulePoseidon::new(poseidon_config);
            // concat values and sk to get the inputs
            let mut inputs = values.iter_mut().map(|x| vec![x.clone()]).collect_vec();
            // layout the module
            Self::layout_module(
                &mut chip,
                layouter,
                &mut inputs,
                &mut instance_offset.poseidon,
            )?;
            // replace the inputs with the outputs
            values.iter_mut().enumerate().for_each(|(i, x)| {
                x.clone_from(&inputs[i][0]);
            });

        // If the module is encrypted, then we need to encrypt the inputs
        } else if element_visibility.is_encrypted() && values.len() > 0 {
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
            Self::layout_module(
                &mut chip,
                layouter,
                &mut inputs,
                &mut instance_offset.elgamal,
            )?;
            // replace the inputs with the outputs
            values.iter_mut().enumerate().for_each(|(i, x)| {
                x.clone_from(&inputs[i][0]);
            });
        }

        Ok(())
    }

    /// Run forward pass
    pub fn forward(
        inputs: &[Tensor<i128>],
    ) -> Result<ModuleForwardResult, Box<dyn std::error::Error>> {
        let mut rng = &mut rand::thread_rng();
        let variables = ElGamalVariables::gen_random(&mut rng);

        let poseidon_hash = inputs.iter().fold(vec![], |mut acc, x| {
            let field_elements = x.iter().map(|x| i128_to_felt::<Fp>(*x)).collect();
            let res = ModulePoseidon::run(field_elements).unwrap()[0].clone();
            acc.extend(res.clone());
            acc
        });

        let elgamal_outputs = inputs.iter().fold(vec![], |mut acc: Vec<Vec<Fp>>, x| {
            let field_elements = x.iter().map(|x| i128_to_felt::<Fp>(*x)).collect();
            let ciphers = ElGamalGadget::run((field_elements, variables.clone()))
                .unwrap()
                .clone();

            if acc.is_empty() {
                ciphers
            } else {
                for i in 0..acc.len() {
                    acc[i].extend(ciphers[i].clone());
                }
                acc
            }
        });

        Ok(ModuleForwardResult {
            poseidon_hash: Some(poseidon_hash),
            elgmal_results: Some(ElGamalResult {
                variables,
                ciphertexts: elgamal_outputs,
            }),
        })
    }
}
