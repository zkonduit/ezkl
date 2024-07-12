use crate::circuit::modules::polycommit::{PolyCommitChip, PolyCommitConfig};
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::circuit::region::ConstantsMap;
use crate::tensor::{Tensor, ValTensor};
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Column, ConstraintSystem, Error, Instance, VerifyingKey};
use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2curves::bn256::{Fr as Fp, G1Affine};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::errors::GraphError;
use super::{VarVisibility, Visibility};

/// poseidon len to hash in tree
pub const POSEIDON_LEN_GRAPH: usize = 32;
/// Poseidon number of instances
pub const POSEIDON_INSTANCES: usize = 1;

/// Poseidon module type
pub type ModulePoseidon =
    PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>;
/// Poseidon module config
pub type ModulePoseidonConfig = PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>;

///
#[derive(Clone, Debug, Default)]
pub struct ModuleConfigs {
    /// PolyCommit
    polycommit: Vec<PolyCommitConfig>,
    /// Poseidon
    poseidon: Option<ModulePoseidonConfig>,
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

        for size in module_size.polycommit {
            config
                .polycommit
                .push(PolyCommitChip::configure(cs, (logrows, size)));
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

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModuleForwardResult {
    /// The inputs of the forward pass for poseidon
    pub poseidon_hash: Option<Vec<Fp>>,
    /// The outputs of the forward pass for PolyCommit
    pub polycommit: Option<Vec<Vec<G1Affine>>>,
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
        } else {
            vec![]
        }
    }

    /// get instances
    pub fn get_instances(&self) -> Vec<Vec<Fp>> {
        if let Some(poseidon) = &self.poseidon_hash {
            poseidon.iter().map(|x| vec![*x]).collect()
        } else {
            vec![]
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
///
pub struct ModuleSizes {
    polycommit: Vec<usize>,
    poseidon: (usize, Vec<usize>),
}

impl ModuleSizes {
    /// Create new module sizes
    pub fn new() -> Self {
        ModuleSizes {
            polycommit: vec![],
            poseidon: (
                0,
                vec![0; crate::circuit::modules::poseidon::NUM_INSTANCE_COLUMNS],
            ),
        }
    }

    /// Get the number of constraints
    pub fn max_constraints(&self) -> usize {
        self.poseidon.0
    }
    /// Get the number of instances
    pub fn num_instances(&self) -> Vec<usize> {
        // concat
        self.poseidon.1.clone()
    }
}

/// Graph modules that can process inputs, params and outputs beyond the basic operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphModules {
    polycommit_idx: usize,
}
impl GraphModules {
    ///
    pub fn new() -> GraphModules {
        GraphModules { polycommit_idx: 0 }
    }

    ///
    pub fn reset_index(&mut self) {
        self.polycommit_idx = 0;
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
                if visibility.is_polycommit() {
                    // 1 constraint for each polycommit commitment
                    sizes.polycommit.push(total_len);
                } else if visibility.is_hashed() {
                    sizes.poseidon.0 += ModulePoseidon::num_rows(total_len);
                    // 1 constraints for hash
                    sizes.poseidon.1[0] += 1;
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
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<(), Error> {
        // reserve module 0 for ... modules
        // hash the input and replace the constrained cells in the input
        let cloned_x = (*x).clone();
        x[0] = module
            .layout(layouter, &cloned_x, instance_offset.to_owned(), constants)
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
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<(), Error> {
        if element_visibility.is_polycommit() && !values.is_empty() {
            // concat values and sk to get the inputs
            let mut inputs = values.iter_mut().map(|x| vec![x.clone()]).collect_vec();

            // layout the module
            inputs.iter_mut().for_each(|x| {
                // create the module
                let chip = PolyCommitChip::new(configs.polycommit[self.polycommit_idx].clone());
                // reserve module 2 onwards for polycommit modules
                let module_offset = 3 + self.polycommit_idx;
                layouter
                    .assign_region(|| format!("_enter_module_{}", module_offset), |_| Ok(()))
                    .unwrap();
                Self::layout_module(&chip, layouter, x, instance_offset, constants).unwrap();
                // increment the current index
                self.polycommit_idx += 1;
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
                    Self::layout_module(&chip, layouter, x, instance_offset, constants).unwrap();
                });
                // replace the inputs with the outputs
                values.iter_mut().enumerate().for_each(|(i, x)| {
                    x.clone_from(&inputs[i][0]);
                });
            } else {
                log::error!("Poseidon config not initialized");
                return Err(Error::Synthesis);
            }
            // If the module is encrypted, then we need to encrypt the inputs
        }

        Ok(())
    }

    /// Run forward pass
    pub fn forward<Scheme: CommitmentScheme<Scalar = Fp, Curve = G1Affine>>(
        inputs: &[Tensor<Scheme::Scalar>],
        element_visibility: &Visibility,
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&Scheme::ParamsProver>,
    ) -> Result<ModuleForwardResult, GraphError> {
        let mut poseidon_hash = None;
        let mut polycommit = None;

        if element_visibility.is_hashed() {
            let field_elements = inputs.iter().fold(vec![], |mut acc, x| {
                let res = ModulePoseidon::run(x.to_vec()).unwrap()[0].clone();
                acc.extend(res);
                acc
            });
            poseidon_hash = Some(field_elements);
        }

        if element_visibility.is_polycommit() {
            if let Some(vk) = vk {
                if let Some(srs) = srs {
                    let commitments = inputs.iter().fold(vec![], |mut acc, x| {
                        let res = PolyCommitChip::commit::<Scheme>(
                            x.to_vec(),
                            (vk.cs().blinding_factors() + 1) as u32,
                            srs,
                        );
                        acc.push(res);
                        acc
                    });
                    polycommit = Some(commitments);
                } else {
                    log::warn!("no srs provided for polycommit. processed value will be none");
                }
            } else {
                log::debug!(
                    "no verifying key provided for polycommit. processed value will be none"
                );
            }
        }

        Ok(ModuleForwardResult {
            poseidon_hash,
            polycommit,
        })
    }
}
