use super::*;
use crate::tensor::{ValTensor, VarTensor};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region},
    plonk::{Assigned, Expression, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

/// A type which wraps a VarTensor and provides method for laying it out and assigning to it within a circuit.
/// This could represent for instance a kernel of weight parameters, a bias vector, input data, or output variables we want to constrain
/// within a circuit.
#[derive(Debug, Clone)]
pub struct IOConfig<F: FieldExt + TensorType> {
    pub kernel: VarTensor,
    marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> IOConfig<F> {
    pub fn configure(kernel: VarTensor) -> Self {
        Self {
            kernel,
            marker: PhantomData,
        }
    }

    /// Retrieve the values represented within the columns of the VarTensor (recall that VarTensor
    /// is a Tensor of columns).
    pub fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>> {
        let mut t = match &self.kernel {
            // when fixed we have 1 col per param
            VarTensor::Fixed { inner: f, dims: _ } => {
                f.map(|c| meta.query_fixed(c, Rotation(offset as i32)))
            }
            // when advice we have 1 col per row
            VarTensor::Advice { inner: a, dims: d } => a
                .map(|column| {
                    Tensor::from(
                        (0..*d.last().unwrap())
                            .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                    )
                })
                .combine(),
        };
        t.reshape(self.kernel.dims());
        t
    }

    /// Retrieve the value represented at a specific index within the columns of the VarTensor (recall that VarTensor
    /// is a Tensor of columns).
    pub fn query_idx(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        match &self.kernel {
            VarTensor::Fixed { inner: f, dims: _ } => {
                meta.query_fixed(f[idx], Rotation(offset as i32))
            }
            VarTensor::Advice { inner: a, dims: _ } => {
                meta.query_advice(a[idx], Rotation(offset as i32))
            }
        }
    }

    /// Assigns specific values to the inner VarTensor (kernel attribute).
    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        values: ValTensor<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        match values {
            ValTensor::Value { inner: v, dims: _ } => {
                v.mc_enum_map(|coord, k| match &self.kernel {
                    VarTensor::Fixed { inner: f, dims: _ } => region
                        .assign_fixed(|| "k", f.get(&coord), offset, || k.into())
                        .unwrap(),
                    VarTensor::Advice { inner: a, dims: _ } => {
                        let coord = format_advice_coord(coord);
                        let last = coord.len() - 1;
                        region
                            .assign_advice(
                                || "k",
                                a.get(&coord[0..last]),
                                offset + coord[last],
                                || k.into(),
                            )
                            .unwrap()
                    }
                })
            }
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                v.mc_enum_map(|coord, x| match &self.kernel {
                    VarTensor::Fixed { inner: _, dims: _ } => panic!("not implemented"),
                    VarTensor::Advice { inner: a, dims: _ } => {
                        let coord = format_advice_coord(coord);
                        let last = coord.len() - 1;
                        x.copy_advice(|| "k", region, a.get(&coord[0..last]), offset + coord[last])
                            .unwrap()
                    }
                })
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => {
                v.mc_enum_map(|coord, k| match &self.kernel {
                    VarTensor::Fixed { inner: f, dims: _ } => region
                        .assign_fixed(|| "k", f.get(&coord), offset, || k)
                        .unwrap(),
                    VarTensor::Advice { inner: a, dims: _ } => {
                        let coord = format_advice_coord(coord);
                        let last = coord.len() - 1;
                        region
                            .assign_advice(
                                || "k",
                                a.get(&coord[0..last]),
                                offset + coord[last],
                                || k.into(),
                            )
                            .unwrap()
                    }
                })
            }
        }
    }

}

fn format_advice_coord(coord: &[usize]) -> Vec<usize> {
    let last = coord.len() - 1;
    let mut v = coord.to_vec();
    if last == 0 {
        v.insert(0, 0);
    }
    v
}
