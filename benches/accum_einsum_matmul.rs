use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use ezkl::circuit::einsum::analysis::analyze_einsum_usage;
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::pfsys::create_keys;
use ezkl::pfsys::srs::gen_srs;
use ezkl::tensor::*;
use halo2_proofs::circuit::floor_planner::V1;
use halo2_proofs::plonk::create_proof;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use rand::rngs::OsRng;
use std::collections::HashMap;
use std::marker::PhantomData;

static mut LEN: usize = 4;
static mut K: usize = 15;

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
        let mut config = Self::Config::default();

        let mut equations = HashMap::new();
        equations.insert(params.equation, params.input_axes_to_dims);
        let analysis = analyze_einsum_usage(&equations).unwrap();
        let num_einsum_inner_cols = 2;
        unsafe {
            config
                .configure_einsums(cs, &analysis, num_einsum_inner_cols, K)
                .unwrap();
        }

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

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let mut config = Self::Config::default();

        let default_params = Self::Params::default();

        let mut equations = HashMap::new();
        equations.insert(default_params.equation, default_params.input_axes_to_dims);
        let analysis = analyze_einsum_usage(&equations).unwrap();
        let num_einsum_inner_cols = 1;
        unsafe {
            config
                .configure_einsums(cs, &analysis, num_einsum_inner_cols, K)
                .unwrap();
        }

        config
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

fn runmatmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_einsum_matmul");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(10);
    let len = 512;
    unsafe {
        LEN = len;
    }
    for k in 19..20 {
        let params = unsafe {
            K = k;
            gen_srs::<KZGCommitmentScheme<_>>(K as u32)
        };

        let mut a = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        a.reshape(&[len, len]).unwrap();

        let mut b = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        b.reshape(&[len, len]).unwrap();

        let einsum = Einsum::<Fr>::new("ij,jk->ik", &[&a, &b]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            einsum,
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", k), &k, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit<Fr>>(&circuit, &params, true)
                    .unwrap();
            });
        });

        let pk = create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit<Fr>>(&circuit, &params, false)
            .unwrap();

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_| {
            b.iter(|| {
                let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

                create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                    &params,
                    &pk,
                    &[circuit.clone()],
                    &[&[]],
                    OsRng,
                    &mut transcript,
                )
                .expect("proof generation should not fail");

                transcript.finalize();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, runmatmul);
criterion_main!(benches);
