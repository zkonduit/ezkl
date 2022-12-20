use crate::abort;
use crate::circuit::eltwise::{DivideBy, EltwiseTable, ReLu, Sigmoid};
use crate::commands::Cli;
use crate::tensor::TensorType;
use crate::tensor::{ValTensor, VarTensor};
use clap::Parser;
use halo2_proofs::{arithmetic::FieldExt, plonk::ConstraintSystem};
use itertools::Itertools;
use log::error;
use serde::Deserialize;
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug, Deserialize)]
pub enum Visibility {
    Private,
    Public,
}
impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(&self, Visibility::Public)
    }
}
impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct VarVisibility {
    pub input: Visibility,
    pub params: Visibility,
    pub output: Visibility,
}
impl std::fmt::Display for VarVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(inputs: {}, params: {}, outputs: {})",
            self.input, self.params, self.output
        )
    }
}

impl VarVisibility {
    pub fn from_args() -> Self {
        let args = Cli::parse();

        let input_vis = if args.public_inputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let params_vis = if args.public_params {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let output_vis = if args.public_outputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        if !output_vis.is_public() & !params_vis.is_public() & !input_vis.is_public() {
            abort!("at least one set of variables should be public");
        }
        Self {
            input: input_vis,
            params: params_vis,
            output: output_vis,
        }
    }
}

pub enum TableTypes<F: FieldExt + TensorType> {
    ReLu(Rc<RefCell<EltwiseTable<F, ReLu<F>>>>),
    DivideBy(Rc<RefCell<EltwiseTable<F, DivideBy<F>>>>),
    Sigmoid(Rc<RefCell<EltwiseTable<F, Sigmoid<F>>>>),
}
impl<F: FieldExt + TensorType> TableTypes<F> {
    pub fn get_relu(&self) -> Rc<RefCell<EltwiseTable<F, ReLu<F>>>> {
        match self {
            TableTypes::ReLu(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
    pub fn get_div(&self) -> Rc<RefCell<EltwiseTable<F, DivideBy<F>>>> {
        match self {
            TableTypes::DivideBy(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
    pub fn get_sig(&self) -> Rc<RefCell<EltwiseTable<F, Sigmoid<F>>>> {
        match self {
            TableTypes::Sigmoid(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
}

#[derive(Clone)]
pub struct ModelVars<F: FieldExt + TensorType> {
    pub advices: Vec<VarTensor>,
    pub fixed: Vec<VarTensor>,
    pub instances: Vec<ValTensor<F>>,
}
/// A wrapper for holding all columns that will be assigned to by a model.
impl<F: FieldExt + TensorType> ModelVars<F> {
    pub fn new(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        advice_dims: (usize, usize),
        fixed_dims: (usize, usize),
        instance_dims: (usize, Vec<Vec<usize>>),
    ) -> Self {
        let tensor_max = Cli::parse().max_rotations;

        let advices = (0..advice_dims.0)
            .map(|_| {
                VarTensor::new_advice(
                    cs,
                    logrows as usize,
                    advice_dims.1,
                    vec![advice_dims.1],
                    true,
                    tensor_max,
                )
            })
            .collect_vec();
        let fixed = (0..fixed_dims.0)
            .map(|_| {
                VarTensor::new_fixed(
                    cs,
                    logrows as usize,
                    fixed_dims.1,
                    vec![fixed_dims.1],
                    true,
                    tensor_max,
                )
            })
            .collect_vec();
        let instances = (0..instance_dims.0)
            .map(|i| ValTensor::new_instance(cs, instance_dims.1[i].clone(), true))
            .collect_vec();
        ModelVars {
            advices,
            fixed,
            instances,
        }
    }
}
