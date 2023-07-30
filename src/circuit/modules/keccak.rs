use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use zkevm_circuits::keccak_circuit::KeccakConfig as InnerConfig;

use super::Module;

pub struct KeccakConfig<F> {
    ///
    pub hash_inputs: Vec<Column<Advice>>,
    ///
    pub instance: Option<Column<Instance>>,
    ///
    pub pow5_config: InnerConfig<F>,
}

impl<F> Module<Fp> for KeccakConfig<F> {
    type Config = todo!();

    type InputAssignments = todo!();

    type RunInputs = todo!();

    fn new(config: Self::Config) -> Self {
        todo!()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        todo!()
    }

    fn name(&self) -> &'static str {
        todo!()
    }

    fn run(input: Self::RunInputs) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        todo!()
    }

    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: &[crate::tensor::ValTensor<Fp>],
    ) -> Result<Self::InputAssignments, Error> {
        todo!()
    }

    fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: &[crate::tensor::ValTensor<Fp>],
        row_offsets: Vec<usize>,
    ) -> Result<crate::tensor::ValTensor<Fp>, Error> {
        todo!()
    }

    fn instance_increment_input(&self) -> Vec<usize> {
        todo!()
    }
}