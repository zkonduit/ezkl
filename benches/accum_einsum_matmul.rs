use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use ezkl::circuit::einsum::analysis::analyze_einsum_usage;
use ezkl::circuit::einsum::circuit_params::SingleEinsumParams;
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::pfsys::srs::gen_srs;
use ezkl::pfsys::{create_keys, create_proof_circuit, TranscriptType};
use ezkl::tensor::*;
use halo2_proofs::circuit::floor_planner::V1;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use rand::rngs::OsRng;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::collections::HashMap;

static mut LEN: usize = 4;
static mut K: usize = 15;

#[derive(Clone)]
struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
    inputs: [ValTensor<F>; 2],
    einsum_params: SingleEinsumParams<F>,
}

impl Circuit<Fr> for MyCircuit<Fr> {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = V1;
    type Params = SingleEinsumParams<Fr>;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fr>, params: Self::Params) -> Self::Config {
        let mut config = Self::Config::default();

        let mut equations = HashMap::new();
        equations.insert((0, params.equation), params.input_axes_to_dims);
        let analysis = analyze_einsum_usage(&equations).unwrap();
        let num_einsum_inner_cols = 2;
        unsafe {
            config
                .configure_einsums(cs, &analysis, num_einsum_inner_cols, K)
                .unwrap();
            let _constant = VarTensor::constant_cols(cs, K, 2, false);
        }

        config
    }

    fn params(&self) -> Self::Params {
        SingleEinsumParams::<Fr>::new(
            &self.einsum_params.equation,
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
            .as_ref()
            .ok_or(Error::Synthesis)?
            .challenges()
            .unwrap()
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
                            equation: self.einsum_params.equation.clone(),
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
    let len = 128;
    unsafe {
        LEN = len;
    }
    for k in 15..16 {
        let params = unsafe {
            K = k;
            gen_srs::<KZGCommitmentScheme<_>>(K as u32)
        };

        let mut a = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        a.reshape(&[len, len]).unwrap();

        let mut b = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        b.reshape(&[len, len]).unwrap();

        let einsum_params = SingleEinsumParams::<Fr>::new("ij,jk->ik", &[&a, &b]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            einsum_params,
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
                let prover = create_proof_circuit::<
                    KZGCommitmentScheme<_>,
                    MyCircuit<Fr>,
                    ProverSHPLONK<_>,
                    VerifierSHPLONK<_>,
                    SingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit.clone(),
                    vec![],
                    &params,
                    &pk,
                    CheckMode::UNSAFE,
                    ezkl::Commitments::KZG,
                    TranscriptType::EVM,
                    None,
                    None,
                );
                prover.unwrap();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runmatmul
}
criterion_main!(benches);
