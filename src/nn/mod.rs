use crate::tensor::*;
use halo2_proofs::{arithmetic::FieldExt, circuit::Layouter, plonk::ConstraintSystem};
/// Structs and methods for configuring and assigning to an affine gate within a Halo2 circuit.
pub mod affine;
/// Structs and methods for configuring and assigning to a convolutional gate within a Halo2 circuit.
pub mod cnvrl;
/// Element-wise operations.
pub mod eltwise;

/// Trait for configuring neural network layers in a Halo2 circuit.
pub trait LayerConfig<F: FieldExt + TensorType> {
    /// Takes in VarTensor input and params, creates a series of operations (gates in Halo2 circuit nomenclature)
    /// using both input and params to produce an output to which we can add equality constraints (for proving).
    /// Produces a layer object with attributes we can then assign to when calling layout().
    fn configure(_meta: &mut ConstraintSystem<F>, variables: &[VarTensor]) -> Self;
    /// Takes in ValTensor inputs and assigns them to the variables created when calling configure().
    fn layout(&self, layouter: &mut impl Layouter<F>, inputs: &[ValTensor<F>]) -> ValTensor<F>;
}

pub enum LayerType<
    F: FieldExt + TensorType,
    const STRIDE: usize,
    const PADDING: usize,
    const BITS: usize,
    NL: eltwise::Nonlinearity<F>,
> {
    Conv(cnvrl::ConvConfig<F, STRIDE, PADDING>),
    Affine(affine::Affine1dConfig<F>),
    Eltwise(eltwise::EltwiseConfig<F, BITS, NL>),
}
