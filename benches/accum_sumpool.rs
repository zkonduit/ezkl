use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::hybrid::HybridOp;
use ezkl::circuit::*;
use ezkl::pfsys::create_keys;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::srs::gen_srs;
use ezkl::pfsys::TranscriptType;
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 3;

const K: usize = 17;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = 10;

        let a = VarTensor::new_advice(cs, K, 1, len * len);

        let b = VarTensor::new_advice(cs, K, 1, len * len);

        let output = VarTensor::new_advice(cs, K, 1, (len + 1) * len);

        Self::Config::configure(cs, &[a, b], &output, CheckMode::UNSAFE)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |region| {
                let mut region = region::RegionCtx::new(region, 0, 1, 1024, 2);
                config
                    .layout(
                        &mut region,
                        &[self.image.clone()],
                        Box::new(HybridOp::SumPool {
                            padding: vec![(0, 0); 2],
                            stride: vec![1, 1],
                            kernel_shape: vec![2, 2],
                            normalized: false,
                            data_format: DataFormat::NCHW,
                        }),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runsumpool(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_sumpool");

    let params = gen_srs::<KZGCommitmentScheme<_>>(K as u32);

    for size in [1, 2].iter() {
        unsafe {
            IMAGE_HEIGHT = size * 4;
            IMAGE_WIDTH = size * 4;

            let mut image = Tensor::from(
                (0..IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
                    .map(|_| Value::known(Fr::random(OsRng))),
            );
            image
                .reshape(&[1, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])
                .unwrap();

            let circuit = MyCircuit {
                image: ValTensor::from(image),
            };

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("pk", size), &size, |b, &_| {
                b.iter(|| {
                    create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true)
                        .unwrap();
                });
            });

            let pk = create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true)
                .unwrap();

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
                b.iter(|| {
                    let prover = create_proof_circuit::<
                        KZGCommitmentScheme<_>,
                        MyCircuit,
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
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runsumpool
}
criterion_main!(benches);
