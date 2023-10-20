use crate::circuit::modules::elgamal::{ElGamalConfig, ElGamalGadget, ElGamalVariables};
use crate::circuit::modules::kzg::{KZGChip, KZGConfig};
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::tensor::{Tensor, ValTensor, ValType};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Column, ConstraintSystem, Error, Instance, VerifyingKey};
use halo2_proofs::poly::kzg::commitment::ParamsKZG;
use halo2curves::bn256::{Bn256, Fr as Fp, G1Affine};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::GraphWitness;
use super::{VarVisibility, Visibility};

/// poseidon len to hash in tree
pub const POSEIDON_LEN_GRAPH: usize = 32;

/// ElGamal number of instances
pub const ELGAMAL_INSTANCES: usize = 4;
/// Poseidon number of instancess
pub const POSEIDON_INSTANCES: usize = 1;

/// Poseidon module type
pub type ModulePoseidon =
    PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>;
/// Poseidon module config
pub type ModulePoseidonConfig = PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>;

///
#[derive(Clone, Debug, Default)]
pub struct ModuleConfigs {
    /// KZG
    kzg: Vec<KZGConfig>,
    /// Poseidon
    poseidon: Option<ModulePoseidonConfig>,
    /// ElGamal
    elgamal: Option<ElGamalConfig>,
    /// Instance
    pub instance: Option<Column<Instance>>,
}

impl ModuleConfigs {
    /// Create new module configs from visibility of each variable
    pub fn from_visibility(
        cs: &mut ConstraintSystem<Fp>,
        module_size: ModuleSizes,
        logrows: usize,
    ) -> Self {
        let mut config = Self::default();

        for size in module_size.kzg {
            config.kzg.push(KZGChip::configure(cs, (logrows, size)));
        }

        config
    }

    /// Configure the modules
    pub fn configure_complex_modules(
        &mut self,
        cs: &mut ConstraintSystem<Fp>,
        visibility: VarVisibility,
        module_size: ModuleSizes,
    ) {
        if (visibility.input.is_encrypted()
            || visibility.output.is_encrypted()
            || visibility.params.is_encrypted())
            && module_size.elgamal.1[0] > 0
        {
            let elgamal = ElGamalGadget::configure(cs, ());
            self.instance = Some(elgamal.instance);
            self.elgamal = Some(elgamal);
        };

        if (visibility.input.is_hashed()
            || visibility.output.is_hashed()
            || visibility.params.is_hashed())
            && module_size.poseidon.1[0] > 0
        {
            if visibility.input.is_hashed_public()
                || visibility.output.is_hashed_public()
                || visibility.params.is_hashed_public()
            {
                if let Some(inst) = self.instance {
                    self.poseidon = Some(ModulePoseidon::configure_with_optional_instance(
                        cs,
                        Some(inst),
                    ));
                } else {
                    let poseidon = ModulePoseidon::configure(cs, ());
                    self.instance = poseidon.instance;
                    self.poseidon = Some(poseidon);
                }
            } else if visibility.input.is_hashed_private()
                || visibility.output.is_hashed_private()
                || visibility.params.is_hashed_private()
            {
                self.poseidon = Some(ModulePoseidon::configure_with_optional_instance(cs, None));
            }
        };
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModuleForwardResult {
    /// The inputs of the forward pass for poseidon
    pub poseidon_hash: Option<Vec<Fp>>,
    /// The outputs of the forward pass for ElGamal
    pub elgamal: Option<ElGamalResult>,
    /// The outputs of the forward pass for KZG
    pub kzg_commit: Option<Vec<Vec<G1Affine>>>,
}

impl ModuleForwardResult {
    /// Get the result
    pub fn get_result(&self, vis: Visibility) -> Vec<Vec<Fp>> {
        if vis.is_hashed() {
            self.poseidon_hash
                .clone()
                .unwrap()
                .into_iter()
                .map(|x| vec![x])
                .collect()
        } else if vis.is_encrypted() {
            self.elgamal.clone().unwrap().encrypted_messages
        } else {
            vec![]
        }
    }

    /// get instances
    pub fn get_instances(&self) -> Vec<Vec<Fp>> {
        if let Some(poseidon) = &self.poseidon_hash {
            poseidon.iter().map(|x| vec![*x]).collect()
        } else if let Some(elgamal) = &self.elgamal {
            elgamal.ciphertexts.clone()
        } else {
            vec![]
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
///
pub struct ModuleSizes {
    kzg: Vec<usize>,
    poseidon: (usize, Vec<usize>),
    elgamal: (usize, Vec<usize>),
}

impl ModuleSizes {
    /// Create new module sizes
    pub fn new() -> Self {
        ModuleSizes {
            kzg: vec![],
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
pub struct GraphModules {
    kzg_idx: usize,
}
impl GraphModules {
    ///
    pub fn new() -> GraphModules {
        GraphModules { kzg_idx: 0 }
    }

    ///
    pub fn reset_index(&mut self) {
        self.kzg_idx = 0;
    }
}

impl GraphModules {
    fn num_constraint_given_shapes(
        visibility: Visibility,
        shapes: Vec<Vec<usize>>,
        sizes: &mut ModuleSizes,
    ) {
        for shape in shapes {
            let total_len = shape.iter().product::<usize>();
            if total_len > 0 {
                if visibility.is_kzgcommit() {
                    // 1 constraint for each kzg commitment
                    sizes.kzg.push(total_len);
                } else if visibility.is_hashed() {
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
        x: &mut Vec<ValTensor<Fp>>,
        instance_offset: &mut usize,
    ) -> Result<(), Error> {
        // reserve module 0 for ... modules
        // hash the input and replace the constrained cells in the input
        let cloned_x = (*x).clone();
        x[0] = module
            .layout(layouter, &cloned_x, instance_offset.to_owned())
            .unwrap();
        for inc in module.instance_increment_input().iter() {
            // increment the instance offset to make way for future module layouts
            *instance_offset += inc;
        }

        Ok(())
    }

    /// Layout the module
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<Fp>,
        configs: &mut ModuleConfigs,
        values: &mut [ValTensor<Fp>],
        element_visibility: &Visibility,
        instance_offset: &mut usize,
        module_settings: &ModuleVarSettings,
    ) -> Result<(), Error> {
        if element_visibility.is_kzgcommit() && !values.is_empty() {
            // concat values and sk to get the inputs
            let mut inputs = values.iter_mut().map(|x| vec![x.clone()]).collect_vec();

            // layout the module
            inputs.iter_mut().for_each(|x| {
                // create the module
                let chip = KZGChip::new(configs.kzg[self.kzg_idx].clone());
                // reserve module 2 onwards for kzg modules
                let module_offset = 3 + self.kzg_idx;
                layouter
                    .assign_region(|| format!("_enter_module_{}", module_offset), |_| Ok(()))
                    .unwrap();
                Self::layout_module(&chip, layouter, x, instance_offset).unwrap();
                // increment the current index
                self.kzg_idx += 1;
            });

            // replace the inputs with the outputs
            values.iter_mut().enumerate().for_each(|(i, x)| {
                x.clone_from(&inputs[i][0]);
            });
        }

        // If the module is hashed, then we need to hash the inputs
        if element_visibility.is_hashed() && !values.is_empty() {
            if let Some(config) = &mut configs.poseidon {
                // reserve module 0 for poseidon modules
                layouter.assign_region(|| "_enter_module_0", |_| Ok(()))?;
                // create the module
                let chip = ModulePoseidon::new(config.clone());
                // concat values and sk to get the inputs
                let mut inputs = values.iter_mut().map(|x| vec![x.clone()]).collect_vec();
                // layout the module
                inputs.iter_mut().for_each(|x| {
                    Self::layout_module(&chip, layouter, x, instance_offset).unwrap();
                });
                // replace the inputs with the outputs
                values.iter_mut().enumerate().for_each(|(i, x)| {
                    x.clone_from(&inputs[i][0]);
                });
            } else {
                panic!("Poseidon config not initialized");
            }
        // If the module is encrypted, then we need to encrypt the inputs
        } else if element_visibility.is_encrypted() && !values.is_empty() {
            if let Some(config) = &mut configs.elgamal {
                // reserve module 1 for elgamal modules
                layouter.assign_region(|| "_enter_module_1", |_| Ok(()))?;
                // create the module
                let mut chip = ElGamalGadget::new(config.clone());
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
                inputs.iter_mut().for_each(|x| {
                    Self::layout_module(&chip, layouter, x, instance_offset).unwrap();
                    chip.config.initialized = true;
                });
                // replace the inputs with the outputs
                values.iter_mut().enumerate().for_each(|(i, x)| {
                    x.clone_from(&inputs[i][0]);
                });

                config.initialized = true;
            } else {
                panic!("ElGamal config not initialized");
            }
        }

        Ok(())
    }

    /// Run forward pass
    pub fn forward(
        inputs: &[Tensor<Fp>],
        element_visibility: Visibility,
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&ParamsKZG<Bn256>>,
    ) -> Result<ModuleForwardResult, Box<dyn std::error::Error>> {
        let mut rng = &mut rand::thread_rng();
        let mut poseidon_hash = None;
        let mut elgamal = None;
        let mut kzg_commit = None;

        if element_visibility.is_hashed() {
            let field_elements = inputs.iter().fold(vec![], |mut acc, x| {
                let res = ModulePoseidon::run(x.to_vec()).unwrap()[0].clone();
                acc.extend(res);
                acc
            });
            poseidon_hash = Some(field_elements);
        }

        if element_visibility.is_kzgcommit() {
            if let Some(vk) = vk {
                if let Some(srs) = srs {
                    let commitments = inputs.iter().fold(vec![], |mut acc, x| {
                        let res = KZGChip::commit(
                            x.to_vec(),
                            vk.cs().degree() as u32,
                            (vk.cs().blinding_factors() + 1) as u32,
                            &srs,
                        );
                        acc.push(res);
                        acc
                    });
                    kzg_commit = Some(commitments);
                } else {
                    log::warn!("no srs provided for kzgcommit. processed value will be none");
                }
            } else {
                log::warn!("no verifying key provided for kzgcommit. processed value will be none");
            }
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
            kzg_commit,
        })
    }
}
