///
pub mod poseidon;

///
pub mod elgamal;

///
pub mod planner;
use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Error},
};
use halo2curves::ff::PrimeField;
pub use planner::*;

use crate::tensor::{TensorType, ValTensor};

/// Module trait used to extend ezkl functionality
pub trait Module<F: PrimeField + TensorType + PartialOrd> {
    /// Config
    type Config;
    /// The return type after an input assignment
    type InputAssignments;
    /// The inputs used in the run function
    type RunInputs;

    /// construct new module from config
    fn new(config: Self::Config) -> Self;
    /// Configure
    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config;
    /// Name
    fn name(&self) -> &'static str;
    /// Run the operation the module represents
    fn run(input: Self::RunInputs) -> Result<Vec<Vec<F>>, Box<dyn std::error::Error>>;
    /// Layout inputs
    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<F>,
        input: &[ValTensor<F>],
    ) -> Result<Self::InputAssignments, Error>;
    /// Layout
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: &[ValTensor<F>],
        row_offsets: Vec<usize>,
    ) -> Result<ValTensor<F>, Error>;
    /// Number of instance values the module uses
    fn instance_increment_input(&self, _: Vec<usize>) -> Vec<usize>;
    /// How to increment when creating a new module
    fn instance_increment_module(&self) -> Vec<usize>;
}