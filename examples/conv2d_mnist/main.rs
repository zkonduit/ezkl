use ezkl::circuit::region::RegionCtx;
use ezkl::circuit::{
    ops::lookup::LookupOp, ops::poly::PolyOp, BaseConfig as PolyConfig, CheckMode,
};
use ezkl::fieldutils;
use ezkl::fieldutils::i32_to_felt;
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Column, ConstraintSystem, Error,
        Instance,
    },
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::bn256::Bn256;
use halo2curves::bn256::Fr as F;

use instant::Instant;
use mnist::*;
use rand::rngs::OsRng;
use std::marker::PhantomData;

mod params;

const K: usize = 20;
const NUM_INNER_COLS: usize = 1;

#[derive(Clone)]
struct Config<
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust stable
    const CLASSES: usize,
    const LOOKUP_MIN: i128,
    const LOOKUP_MAX: i128,
    // Convolution
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const OUT_CHANNELS: usize,
    const STRIDE: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
    const IN_CHANNELS: usize,
    const PADDING: usize,
> where
    Value<F>: TensorType,
{
    // this will be a conv layer
    layer_config: PolyConfig<F>,
    // this will be an affine layer
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const CLASSES: usize,
    const LOOKUP_MIN: i128,
    const LOOKUP_MAX: i128,
    // Convolution
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const OUT_CHANNELS: usize,
    const STRIDE: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
    const IN_CHANNELS: usize,
    const PADDING: usize,
> where
    Value<F>: TensorType,
{
    // Given the stateless ConvConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: ValTensor<F>,
    l0_params: [Tensor<F>; 2],
    l2_params: [Tensor<F>; 2],
}

impl<
        const LEN: usize,
        const CLASSES: usize,
        const LOOKUP_MIN: i128,
        const LOOKUP_MAX: i128,
        // Convolution
        const KERNEL_HEIGHT: usize,
        const KERNEL_WIDTH: usize,
        const OUT_CHANNELS: usize,
        const STRIDE: usize,
        const IMAGE_HEIGHT: usize,
        const IMAGE_WIDTH: usize,
        const IN_CHANNELS: usize,
        const PADDING: usize,
    > Circuit<F>
    for MyCircuit<
        LEN,
        CLASSES,
        LOOKUP_MIN,
        LOOKUP_MAX,
        KERNEL_HEIGHT,
        KERNEL_WIDTH,
        OUT_CHANNELS,
        STRIDE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IN_CHANNELS,
        PADDING,
    >
where
    Value<F>: TensorType,
{
    type Config = Config<
        LEN,
        CLASSES,
        LOOKUP_MIN,
        LOOKUP_MAX,
        KERNEL_HEIGHT,
        KERNEL_WIDTH,
        OUT_CHANNELS,
        STRIDE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IN_CHANNELS,
        PADDING,
    >;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = PhantomData<F>;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let input = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
        let params = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
        let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);

        println!("INPUT COL {:#?}", input);

        let mut layer_config = PolyConfig::configure(
            cs,
            &[input.clone(), params.clone()],
            &output,
            CheckMode::SAFE,
        );

        layer_config
            .configure_lookup(
                cs,
                &input,
                &output,
                &params,
                (LOOKUP_MIN, LOOKUP_MAX),
                K,
                &LookupOp::ReLU,
            )
            .unwrap();

        layer_config
            .configure_lookup(
                cs,
                &input,
                &output,
                &params,
                (LOOKUP_MIN, LOOKUP_MAX),
                K,
                &LookupOp::Div { denom: 32.0.into() },
            )
            .unwrap();

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        Config {
            layer_config,
            public_output,
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.layer_config.layout_tables(&mut layouter).unwrap();

        let x = layouter
            .assign_region(
                || "mlp_4d",
                |region| {
                    let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS);

                    let op = PolyOp::Conv {
                        kernel: self.l0_params[0].clone(),
                        bias: Some(self.l0_params[1].clone()),
                        padding: [(PADDING, PADDING); 2],
                        stride: (STRIDE, STRIDE),
                    };
                    let x = config
                        .layer_config
                        .layout(&mut region, &[self.input.clone()], Box::new(op))
                        .unwrap();

                    let x = config
                        .layer_config
                        .layout(&mut region, &[x.unwrap()], Box::new(LookupOp::ReLU))
                        .unwrap();

                    let mut x = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[x.unwrap()],
                            Box::new(LookupOp::Div { denom: 32.0.into() }),
                        )
                        .unwrap()
                        .unwrap();

                    x.flatten();
                    // multiply by weights
                    let x = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[self.l2_params[0].clone().into(), x],
                            Box::new(PolyOp::Einsum {
                                equation: "ij,j->ik".to_string(),
                            }),
                        )
                        .unwrap()
                        .unwrap();
                    // add bias
                    let x: ValTensor<F> = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[x, self.l2_params[1].clone().into()],
                            Box::new(PolyOp::Add),
                        )
                        .unwrap()
                        .unwrap();
                    Ok(x)
                },
            )
            .unwrap();

        match x {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => v
                .enum_map(|i, x| match x {
                    ValType::PrevAssigned(v) => {
                        layouter.constrain_instance(v.cell(), config.public_output, i)
                    }
                    _ => panic!(),
                })
                .unwrap(),
            _ => panic!("Should be assigned"),
        };

        Ok(())
    }
}

pub fn runconv() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    const KERNEL_HEIGHT: usize = 5;
    const KERNEL_WIDTH: usize = 5;
    const OUT_CHANNELS: usize = 4;
    const STRIDE: usize = 2;
    const IMAGE_HEIGHT: usize = 28;
    const IMAGE_WIDTH: usize = 28;
    const IN_CHANNELS: usize = 1;
    const PADDING: usize = 0;
    const CLASSES: usize = 10;
    const LEN: usize = {
        OUT_CHANNELS
            * ((IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
            * ((IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)
    };

    // Load the parameters and preimage from somewhere

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img: _,
        tst_lbl: _,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut train_data = Tensor::from(trn_img.iter().map(|x| i32_to_felt::<F>(*x as i32 / 16)));
    train_data.reshape(&[50_000, 28, 28]).unwrap();

    let mut train_labels = Tensor::from(trn_lbl.iter().map(|x| *x as f32));
    train_labels.reshape(&[50_000, 1]).unwrap();

    println!("The first digit is a {:?}", train_labels[0]);

    let mut input: ValTensor<F> = train_data
        .get_slice(&[0..1, 0..28, 0..28])
        .unwrap()
        .map(Value::known)
        .into();

    input.reshape(&[1, 1, 28, 28]).unwrap();

    let myparams = params::Params::new();
    let mut l0_kernels = Tensor::<F>::from(
        myparams
            .kernels
            .clone()
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|fl| {
                let dx = fl * 32_f32;
                let rounded = dx.round();
                let integral: i32 = unsafe { rounded.to_int_unchecked() };
                fieldutils::i32_to_felt(integral)
            }),
    );

    l0_kernels
        .reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH])
        .unwrap();
    l0_kernels.set_visibility(&ezkl::graph::Visibility::Private);

    let mut l0_bias = Tensor::<F>::from((0..OUT_CHANNELS).map(|_| fieldutils::i32_to_felt(0)));
    l0_bias.set_visibility(&ezkl::graph::Visibility::Private);

    let mut l2_biases = Tensor::<F>::from(myparams.biases.into_iter().map(|fl| {
        let dx = fl * 32_f32;
        let rounded = dx.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        fieldutils::i32_to_felt(integral)
    }));
    l2_biases.set_visibility(&ezkl::graph::Visibility::Private);
    l2_biases.reshape(&[l2_biases.len(), 1]).unwrap();

    let mut l2_weights = Tensor::<F>::from(myparams.weights.into_iter().flatten().map(|fl| {
        let dx = fl * 32_f32;
        let rounded = dx.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        fieldutils::i32_to_felt(integral)
    }));
    l2_weights.set_visibility(&ezkl::graph::Visibility::Private);
    l2_weights.reshape(&[CLASSES, LEN]).unwrap();

    let circuit = MyCircuit::<
        LEN,
        10,
        -32768,
        32768,
        KERNEL_HEIGHT,
        KERNEL_WIDTH,
        OUT_CHANNELS,
        STRIDE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IN_CHANNELS,
        PADDING,
    > {
        input,
        l0_params: [l0_kernels, l0_bias],
        l2_params: [l2_weights, l2_biases],
    };

    let public_input: Tensor<i32> = vec![
        -25124i32, -19304, -16668, -4399, -6209, -4548, -2317, -8349, -6117, -23461,
    ]
    .into_iter()
    .into();

    let pi_inner: Tensor<F> = public_input.map(i32_to_felt::<F>);

    println!("MOCK PROVING");
    let now = Instant::now();
    let prover = MockProver::run(
        K as u32,
        &circuit,
        vec![pi_inner.clone().into_iter().collect()],
    )
    .unwrap();
    prover.assert_satisfied();
    let elapsed = now.elapsed();
    println!(
        "MOCK PROVING took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner]];

    //	Real proof
    println!("SRS GENERATION");
    let now = Instant::now();
    let params: ParamsKZG<Bn256> = ParamsKZG::new(K as u32);
    let elapsed = now.elapsed();
    println!(
        "SRS GENERATION took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    let empty_circuit = circuit.without_witnesses();

    // Initialize the proving key
    println!("VK GENERATION");
    let now = Instant::now();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let elapsed = now.elapsed();
    println!(
        "VK GENERATION took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    println!("PK GENERATION");
    let now = Instant::now();
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
    let elapsed = now.elapsed();
    println!(
        "PK GENERATION took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    println!("PROOF GENERATION");
    let now = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut rng = OsRng;
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        pi_for_real_prover,
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    let elapsed = now.elapsed();
    println!(
        "PROOF GENERATION took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    let now = Instant::now();
    let strategy = SingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let verify = verify_proof::<_, VerifierSHPLONK<_>, _, _, _>(
        &params,
        pk.get_vk(),
        strategy,
        pi_for_real_prover,
        &mut transcript,
    );
    assert!(verify.is_ok());

    let elapsed = now.elapsed();
    println!(
        "Verify took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
}

fn main() {
    runconv()
}
