use crate::tensor::*;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter},
    plonk::{Assigned, ConstraintSystem},
};

pub mod affine;
pub mod cnvrl;
pub mod io;

pub trait LayerConfig<F: FieldExt + TensorType> {
    fn configure(
        _meta: &mut ConstraintSystem<F>,
        params: &[VarTensor],
        input: VarTensor,
        output: VarTensor,
    ) -> Self;
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        params: &[ValTensor<F>],
    ) -> ValTensor<F>;
    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        params: &[ValTensor<F>],
    ) -> Tensor<AssignedCell<Assigned<F>, F>>;
}
