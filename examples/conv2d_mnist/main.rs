use ezkl_lib::circuit::base::{BaseConfig as PolyConfig, CheckMode, Op as PolyOp};
use ezkl_lib::circuit::lookup::{Config as LookupConfig, Op as LookupOp};
use ezkl_lib::fieldutils;
use ezkl_lib::fieldutils::i32_to_felt;
use ezkl_lib::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Column, ConstraintSystem, Error,
        Instance,
    },
    poly::{
        commitment::ParamsProver,
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::vesta;
use halo2curves::pasta::Fp as F;
use mnist::*;
use rand::rngs::OsRng;
use std::time::Instant;

mod params;

const K: usize = 20;

#[derive(Clone)]
struct Config<
    F: FieldExt + TensorType,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust stable
    const CLASSES: usize,
    const BITS: usize,
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
    relu: LookupConfig<F>,
    // this will be an affine layer
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt + TensorType,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const CLASSES: usize,
    const BITS: usize,
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
    l0_params: [ValTensor<F>; 2],
    l2_params: [ValTensor<F>; 2],
}

impl<
        F: FieldExt + TensorType,
        const LEN: usize,
        const CLASSES: usize,
        const BITS: usize,
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
        F,
        LEN,
        CLASSES,
        BITS,
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
        F,
        LEN,
        CLASSES,
        BITS,
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

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let input = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
        let params = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
        let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

        println!("INPUT COL {:#?}", input);

        let layer_config =
            PolyConfig::configure(cs, &[input.clone(), params], &output, CheckMode::SAFE, 0);

        let input = input.reshape(&[LEN]);
        let output = output.reshape(&[LEN]);

        let relu =
            LookupConfig::configure(cs, &input, &output, BITS, &[LookupOp::ReLU { scale: 32 }]);

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        Config {
            layer_config,
            relu,
            public_output,
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.relu.layout_table(&mut layouter).unwrap();

        let x = layouter
            .assign_region(
                || "mlp_4d",
                |mut region| {
                    let mut offset = 0;
                    let op = PolyOp::Conv {
                        padding: (PADDING, PADDING),
                        stride: (STRIDE, STRIDE),
                    };
                    let x = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[
                                self.input.clone(),
                                self.l0_params[0].clone(),
                                self.l0_params[1].clone(),
                            ],
                            &mut offset,
                            op.clone(),
                        )
                        .unwrap();

                    let mut x = config.relu.layout(&mut region, &x, &mut offset).unwrap();
                    x.flatten();
                    let l2out = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[x, self.l2_params[0].clone(), self.l2_params[1].clone()],
                            &mut offset,
                            PolyOp::Affine,
                        )
                        .unwrap();
                    Ok(l2out)
                },
            )
            .unwrap();

        match x {
            ValTensor::Value { inner: v, dims: _ } => v
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
    train_data.reshape(&[50_000, 28, 28]);

    let mut train_labels = Tensor::from(trn_lbl.iter().map(|x| *x as f32));
    train_labels.reshape(&[50_000, 1]);

    println!("The first digit is a {:?}", train_labels[0]);

    let mut input: ValTensor<F> = train_data
        .get_slice(&[0..1, 0..28, 0..28])
        .unwrap()
        .map(Value::known)
        .into();

    input.reshape(&[1, 28, 28]).unwrap();

    let myparams = params::Params::new();
    let mut l0_kernels: ValTensor<F> = Tensor::<Value<F>>::from(
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
                let felt = fieldutils::i32_to_felt(integral);
                Value::known(felt)
            }),
    )
    .into();

    l0_kernels
        .reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH])
        .unwrap();

    let l0_bias: ValTensor<F> = Tensor::<Value<F>>::from(
        (0..OUT_CHANNELS).map(|_| Value::known(fieldutils::i32_to_felt(0))),
    )
    .into();

    let mut l2_biases: ValTensor<F> =
        Tensor::<Value<F>>::from(myparams.biases.into_iter().map(|fl| {
            let dx = fl * 32_f32;
            let rounded = dx.round();
            let integral: i32 = unsafe { rounded.to_int_unchecked() };
            let felt = fieldutils::i32_to_felt(integral);
            Value::known(felt)
        }))
        .into();

    l2_biases.reshape(&[l2_biases.len(), 1]).unwrap();

    let mut l2_weights: ValTensor<F> =
        Tensor::<Value<F>>::from(myparams.weights.into_iter().flatten().map(|fl| {
            let dx = fl * 32_f32;
            let rounded = dx.round();
            let integral: i32 = unsafe { rounded.to_int_unchecked() };
            let felt = fieldutils::i32_to_felt(integral);
            Value::known(felt)
        }))
        .into();

    l2_weights.reshape(&[CLASSES, LEN]).unwrap();

    let circuit = MyCircuit::<
        F,
        LEN,
        10,
        16,
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

    #[cfg(feature = "dev-graph")]
    {
        println!("Plotting");
        use plotters::prelude::*;

        let root = BitMapBackend::new("conv2dmnist-layout.png", (2048, 7680)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Conv -> ReLU -> Affine -> ReLU", ("sans-serif", 60))
            .unwrap();

        halo2_proofs::dev::CircuitLayout::default()
            .render(13, &circuit, &root)
            .unwrap();
        return;
    }

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
    println!("MOCK PROVING took {}", now.elapsed().as_secs());

    let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner]];

    //	Real proof
    println!("SRS GENERATION");
    let now = Instant::now();
    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(K as u32);
    println!("SRS GENERATION took {}", now.elapsed().as_secs());
    let empty_circuit = circuit.without_witnesses();
    // Initialize the proving key
    println!("VK GENERATION");
    let now = Instant::now();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    println!("VK GENERATION took {}", now.elapsed().as_secs());
    println!("PK GENERATION");
    let now = Instant::now();
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
    println!("PK GENERATION took {}", now.elapsed().as_secs());
    println!("PROOF GENERATION");
    let now = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut rng = OsRng;
    create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        pi_for_real_prover,
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    //println!("{:?}", proof);
    println!("PROOF GENERATION took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let strategy = SingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(verify_proof(
        &params,
        pk.get_vk(),
        strategy,
        pi_for_real_prover,
        &mut transcript
    )
    .is_ok());
    println!("Verify took {}", now.elapsed().as_secs());
}

fn main() {
    runconv()
}
