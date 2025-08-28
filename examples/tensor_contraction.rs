use ezkl::circuit::einsum::analysis::analyze_einsum_usage;
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::tensor::*;
use halo2_proofs::circuit::floor_planner::V1;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use rand::rngs::OsRng;
use std::collections::HashMap;
use std::marker::PhantomData;

static mut LEN: usize = 4;
const K: usize = 11;

#[derive(Clone)]
struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
    inputs: [ValTensor<F>; 2],
    einsum: Einsum<F>,
}

#[derive(Clone, Default)]
struct Einsum<F: PrimeField + TensorType + PartialOrd> {
    equation: String,
    input_axes_to_dims: HashMap<char, usize>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> Einsum<F> {
    pub fn new(equation: &str, inputs: &[&Tensor<Value<F>>]) -> Result<Self, CircuitError> {
        let mut eq = equation.split("->");
        let inputs_eq = eq.next().ok_or(CircuitError::InvalidEinsum)?;
        let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

        // Check that the number of inputs matches the number of inputs in the equation
        if inputs.len() != inputs_eq.len() {
            return Err(TensorError::DimMismatch("einsum".to_string()).into());
        }

        let mut input_axes_to_dims = HashMap::new();
        for (i, input) in inputs.iter().enumerate() {
            for j in 0..inputs_eq[i].len() {
                let c = inputs_eq[i]
                    .chars()
                    .nth(j)
                    .ok_or(CircuitError::InvalidEinsum)?;
                if let std::collections::hash_map::Entry::Vacant(e) = input_axes_to_dims.entry(c) {
                    e.insert(input.dims()[j]);
                } else if input_axes_to_dims[&c] != input.dims()[j] {
                    return Err(TensorError::DimMismatch("einsum".to_string()).into());
                }
            }
        }

        Ok(Self {
            equation: equation.to_owned(),
            input_axes_to_dims,
            _marker: PhantomData,
        })
    }
}

impl Circuit<Fr> for MyCircuit<Fr> {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = V1;
    type Params = Einsum<Fr>;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fr>, params: Self::Params) -> Self::Config {
        let len = unsafe { LEN };

        let a = VarTensor::new_advice(cs, K, 1, len);
        let b = VarTensor::new_advice(cs, K, 1, len);
        let output = VarTensor::new_advice(cs, K, 1, len);

        let mut config = Self::Config::configure(cs, &[a, b], &output, CheckMode::UNSAFE);

        let mut equations = HashMap::new();
        equations.insert((0, params.equation), params.input_axes_to_dims);
        let analysis = analyze_einsum_usage(&equations).unwrap();
        let num_einsum_inner_cols = 2;
        config
            .configure_einsums(cs, &analysis, num_einsum_inner_cols, K)
            .unwrap();
        let _constant = VarTensor::constant_cols(cs, K, 2, false);

        config
    }

    fn params(&self) -> Self::Params {
        Einsum::<Fr>::new(
            &self.einsum.equation,
            &[
                &self.inputs[0].get_inner().unwrap(),
                &self.inputs[1].get_inner().unwrap(),
            ],
        )
        .unwrap()
    }

    fn configure(_cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        unimplemented!("call configure_with_params instead")
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let challenges = config
            .einsums
            .challenges()
            .iter()
            .map(|c| layouter.get_challenge(*c))
            .collect_vec();

        layouter.assign_region(
            || "",
            |region| {
                let mut region = region::RegionCtx::new_with_challenges(
                    region,
                    0,
                    1,
                    1024,
                    2,
                    challenges.clone(),
                );
                config
                    .layout(
                        &mut region,
                        &self.inputs.iter().collect_vec(),
                        Box::new(PolyOp::Einsum {
                            equation: self.einsum.equation.clone(),
                        }),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runmatmul() {
    let i = 10;
    let n = 10;
    let j = 40;
    let k = 10;

    let mut a = Tensor::from((0..i * n * j).map(|_| Value::known(Fr::random(OsRng))));
    a.reshape(&[i, n, j]).unwrap();

    // parameters
    let mut b = Tensor::from((0..j * k).map(|_| Value::known(Fr::random(OsRng))));
    b.reshape(&[j, k]).unwrap();

    let einsum = Einsum::<Fr>::new("inj,jk->ik", &[&a, &b]).unwrap();

    let circuit = MyCircuit {
        inputs: [ValTensor::from(a), ValTensor::from(b)],
        einsum,
    };

    let mock_prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
    mock_prover.assert_satisfied();
}

pub fn main() {
    runmatmul()
}
