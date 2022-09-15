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
use halo2deeplearning::fieldutils;
use halo2deeplearning::fieldutils::i32tofelt;
use halo2deeplearning::nn::affine::Affine1dConfig;
use halo2deeplearning::nn::cnvrl;
use halo2deeplearning::nn::*;
use halo2deeplearning::tensor::{Tensor, TensorType};
use halo2deeplearning::tensor_ops::eltwise::{DivideBy, NonlinConfig1d, ReLu};
use mnist::*;
use rand::rngs::OsRng;
use std::cmp::max;
use std::time::Instant;

mod params;

#[derive(Clone)]
struct ConvConfig<
    F: FieldExt + TensorType,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust stable
    const CLASSES: usize,
    const INBITS: usize,
    const OUTBITS: usize,
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
    l0: cnvrl::Config<
        F,
        KERNEL_HEIGHT,
        KERNEL_WIDTH,
        OUT_CHANNELS,
        STRIDE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IN_CHANNELS,
        PADDING,
    >,
    l0q: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 32>>,
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, CLASSES>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const CLASSES: usize,
    const INBITS: usize,
    const OUTBITS: usize,
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
        const INBITS: usize,
        const OUTBITS: usize,
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
        INBITS,
        OUTBITS,
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
    type Config = ConvConfig<
        F,
        LEN,
        CLASSES,
        INBITS,
        OUTBITS,
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
        let _output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
        let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

        let num_advices = max(max(output_width, IMAGE_WIDTH), LEN + 3);

        let advices = Tensor::from((0..num_advices).map(|_| cs.advice_column()));
        advices.map(|col| cs.enable_equality(col));

        let mut kernel = Tensor::from((0..KERNEL_WIDTH * KERNEL_HEIGHT).map(|_| cs.fixed_column()));
        kernel.reshape(&[KERNEL_WIDTH, KERNEL_HEIGHT]);

        let l0 = cnvrl::Config::<
            F,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        >::configure(
            cs,
            &[ParamType::Fixed(kernel)],
            ParamType::Advice(advices.clone()),
            ParamType::Advice(advices.clone()),
        );
        let l0q: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 32>> =
            NonlinConfig1d::configure(cs, advices[..LEN].try_into().unwrap());
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, advices[..LEN].try_into().unwrap());

        let l2: Affine1dConfig<F, LEN, CLASSES> = Affine1dConfig::configure(
            cs,
            &[
                ParamType::Advice(advices.get_slice(&[0..CLASSES])),
                ParamType::Advice(advices.get_slice(&[LEN + 2..LEN + 3])),
            ],
            ParamType::Advice(advices.get_slice(&[LEN..LEN + 1])),
            ParamType::Advice(advices.get_slice(&[CLASSES + 1..CLASSES + 2])),
        );
        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);
        //        cs.enable_equality(l2.output);

        ConvConfig {
            l0,
            l0q,
            l1,
            l2,
            // l3,
            // l4,
            public_output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let l0out = config.l0.layout(
            &mut layouter,
            IOType::Value(self.input.clone()),
            &[IOType::Value(self.l0_params.clone())],
        );
        let l0qout = config.l0q.layout(&mut layouter, l0out)?;
        let mut l1out = config.l1.layout(&mut layouter, l0qout)?;
        l1out.reshape(&[1, l1out.dims()[0]]);
        let l2out = config.l2.layout(
            &mut layouter,
            IOType::PrevAssigned(l1out),
            &self
                .l2_params
                .iter()
                .map(|a| IOType::Value(a.clone().into()))
                .collect::<Vec<IOType<F>>>(),
        );

        // tie the last output to public inputs (instance column)
        l2out.enum_map(|i, a| {
            layouter
                .constrain_instance(a.cell(), config.public_output, i)
                .unwrap()
        });

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

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let mut train_data = Tensor::from(trn_img.iter().map(|x| i32tofelt::<F>(*x as i32 / 16)));
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
                let felt = fieldutils::i32tofelt(integral);
                Value::known(felt)
            }),
    );
    // tensorflow is in KHxKWxINxOUT we are OUTxINxWxH?

    kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_WIDTH, KERNEL_HEIGHT]);

    let l2biases = Tensor::<i32>::from(myparams.biases.into_iter().map(|fl| {
        let dx = fl * (32 as f32);
        let rounded = dx.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        integral
    }));

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

    let pi_inner: Tensor<F> = public_input.map(|x| i32tofelt::<F>(x).into());
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
