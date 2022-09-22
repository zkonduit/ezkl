use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Column, ConstraintSystem, Error,
        Fixed, Instance,
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
use halo2deeplearning::fieldutils;
use halo2deeplearning::fieldutils::i32_to_felt;
use halo2deeplearning::nn::affine::Affine1dConfig;
use halo2deeplearning::nn::cnvrl::ConvConfig;
use halo2deeplearning::nn::*;
use halo2deeplearning::tensor::*;
use halo2deeplearning::tensor_ops::eltwise::{DivideBy, EltwiseConfig, ReLu};
use mnist::*;
use rand::rngs::OsRng;
use std::cmp::max;
use std::time::Instant;

mod params;

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
    l0: ConvConfig<F, STRIDE, PADDING>,
    l0q: EltwiseConfig<F, BITS, DivideBy<F, 32>>,
    l1: EltwiseConfig<F, BITS, ReLu<F>>,
    l2: Affine1dConfig<F>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt,
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
    input: Tensor<Value<F>>,
    l0_params: Tensor<Value<F>>,
    l2_params: [Tensor<i32>; 2],
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
        let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
        let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

        let num_advices = max(
            max(output_height * OUT_CHANNELS, IMAGE_HEIGHT * IN_CHANNELS),
            LEN + 3,
        );

        let advices = VarTensor::from(Tensor::from((0..num_advices).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        })));

        let mut kernel: Tensor<Column<Fixed>> =
            (0..OUT_CHANNELS * IN_CHANNELS * KERNEL_WIDTH * KERNEL_HEIGHT)
                .map(|_| cs.fixed_column())
                .into();
        kernel.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

        let l0 = ConvConfig::<F, STRIDE, PADDING>::configure(
            cs,
            &[VarTensor::from(kernel)],
            advices.get_slice(
                &[0..IMAGE_HEIGHT * IN_CHANNELS],
                &[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            ),
            advices.get_slice(
                &[0..output_height * OUT_CHANNELS],
                &[OUT_CHANNELS, output_height, output_width],
            ),
        );

        let l0q: EltwiseConfig<F, BITS, DivideBy<F, 32>> =
            EltwiseConfig::configure(cs, advices.get_slice(&[0..LEN], &[LEN]), None);
        let l1: EltwiseConfig<F, BITS, ReLu<F>> =
            EltwiseConfig::configure(cs, advices.get_slice(&[0..LEN], &[LEN]), None);

        let l2: Affine1dConfig<F> = Affine1dConfig::configure(
            cs,
            &[
                advices.get_slice(&[0..CLASSES], &[CLASSES, LEN]),
                advices.get_slice(&[LEN + 2..LEN + 3], &[CLASSES]),
            ],
            advices.get_slice(&[LEN..LEN + 1], &[LEN]),
            advices.get_slice(&[CLASSES + 1..CLASSES + 2], &[CLASSES]),
        );
        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        Config {
            l0,
            l0q,
            l1,
            l2,
            public_output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let x: Tensor<Value<F>> = self.input.clone().into();
        let l0out = config.l0.layout(
            &mut layouter,
            ValTensor::from(x),
            &[ValTensor::from(self.l0_params.clone())],
        );
        let l0qout = config.l0q.layout(&mut layouter, l0out);
        let mut l1out = config.l1.layout(&mut layouter, l0qout);
        l1out.flatten();
        let l2out = config.l2.layout(
            &mut layouter,
            l1out,
            &self
                .l2_params
                .iter()
                .map(|a| ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(a.clone())))
                .collect::<Vec<ValTensor<F>>>(),
        );

        match l2out {
            ValTensor::PrevAssigned { inner: v, dims: _ } => v.enum_map(|i, x| {
                layouter
                    .constrain_instance(x.cell(), config.public_output, i)
                    .unwrap()
            }),
            _ => panic!("Should be assigned"),
        };

        Ok(())
    }
}

pub fn runconv() {
    const K: u32 = 17;

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

    let mut image = train_data
        .get_slice(&[0..1, 0..28, 0..28])
        .map(|d| Value::known(d));

    image.reshape(&[1, 28, 28]);

    let myparams = params::Params::new();
    let mut kernels = Tensor::from(
        myparams
            .kernels
            .clone()
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|fl| {
                let dx = (fl as f32) * (32 as f32);
                let rounded = dx.round();
                let integral: i32 = unsafe { rounded.to_int_unchecked() };
                let felt = fieldutils::i32_to_felt(integral);
                Value::known(felt)
            }),
    );
    // tensorflow is in KHxKWxINxOUT we are OUTxINxWxH?

    kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

    let l2biases = Tensor::<i32>::from(myparams.biases.into_iter().map(|fl| {
        let dx = fl * (32 as f32);
        let rounded = dx.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        integral
    }));

    // l2biases.reshape(&[1, CLASSES]);

    let mut l2weights = Tensor::<i32>::from(myparams.weights.into_iter().flatten().map(|fl| {
        let dx = fl * (32 as f32);
        let rounded = dx.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        integral
    }));

    l2weights.reshape(&[CLASSES, LEN]);

    let input = image;
    let l0_params = kernels;
    let l2_params = [l2weights, l2biases];

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
        l0_params,
        l2_params,
    };

    let public_input: Tensor<i32> = vec![
        -25124i32, -19304, -16668, -4399, -6209, -4548, -2317, -8349, -6117, -23461,
    ]
    .into_iter()
    .into();

    let pi_inner: Tensor<F> = public_input.map(|x| i32_to_felt::<F>(x).into());
    let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner]];

    //	Real proof
    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(K);
    let empty_circuit = circuit.without_witnesses();
    // Initialize the proving key
    let now = Instant::now();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    println!("VK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
    println!("PK took {}", now.elapsed().as_secs());
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
    println!("Proof took {}", now.elapsed().as_secs());
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
