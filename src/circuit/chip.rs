use super::*;
use halo2_proofs::{
    arithmetic::FieldExt,
    plonk::{Column, ConstraintSystem, Instance},
};
use itertools::Itertools;
pub struct Chip {
    pub advices: Vec<VarTensor>,
    pub fixed: Vec<VarTensor>,
    // TODO: create a new VarTensor for Instance Columns
    pub instances: Vec<Column<Instance>>,
}

impl Chip {
    pub fn new<F: FieldExt>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        advice_dims: (usize, usize),
        fixed_dims: (usize, usize),
        num_instances: usize,
    ) -> Self {
        let advices = (0..advice_dims.0)
            .map(|_| {
                VarTensor::new_advice(
                    cs,
                    logrows as usize,
                    advice_dims.1,
                    vec![advice_dims.1],
                    true,
                )
            })
            .collect_vec();
        // todo init fixed
        let fixed = (0..fixed_dims.0)
            .map(|_| {
                VarTensor::new_fixed(cs, logrows as usize, fixed_dims.1, vec![fixed_dims.1], true)
            })
            .collect_vec();
        let instances = (0..num_instances)
            .map(|_| {
                let l = cs.instance_column();
                cs.enable_equality(l);
                l
            })
            .collect_vec();
        Chip {
            advices,
            fixed,
            instances,
        }
    }
}
