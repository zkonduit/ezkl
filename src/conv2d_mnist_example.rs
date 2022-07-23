use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Error, Instance, SingleVerifier,
    },
    poly::{commitment::Params, Rotation},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};
// use rand::rngs::OsRng;
// use std::marker::PhantomData;

use crate::fieldutils::{felt_to_i32, i32tofelt};
//use crate::tensorutils::{dot3, flatten3, flatten4, map2, map3, map3r, map4, map4r};

use std::cmp::max;

use crate::affine1d::{Affine1d, Affine1dConfig};
use crate::cnvrl_generic;
use crate::eltwise::{DivideBy, Nonlin1d, NonlinConfig1d, ReLu};

#[derive(Clone)]
struct MyConfig<
    F: FieldExt,
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
    [(); (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1]:,
    [(); (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1]:,
    [(); LEN + 3]:,
{
    l0: cnvrl_generic::Config<
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
> {
    // Given the stateless MyConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: [cnvrl_generic::Image<Value<F>, IMAGE_HEIGHT, IMAGE_WIDTH>; IN_CHANNELS],
    l0_params:
        [[cnvrl_generic::Kernel<Value<F>, KERNEL_HEIGHT, KERNEL_WIDTH>; IN_CHANNELS]; OUT_CHANNELS],
    l2_params: (Vec<Vec<i32>>, Vec<i32>),
}

impl<
        F: FieldExt,
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
    [(); (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1]:,
    [(); (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1]:,
    [(); IMAGE_HEIGHT * IMAGE_WIDTH]:,
    [(); ((IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
        * ((IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)]:,
    [(); LEN + 3]:,
{
    type Config = MyConfig<
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
        // Self {
        //     l0: cnvrl_generic::Conv2dAssigned::<
        //         F,
        //         KERNEL_HEIGHT,
        //         KERNEL_WIDTH,
        //         OUT_CHANNELS,
        //         STRIDE,
        //         IMAGE_HEIGHT,
        //         IMAGE_WIDTH,
        //         IN_CHANNELS,
        //         PADDING,
        //     >::without_witnesses(),

        //     //            l1: Nonlin1d::<F, Value<Assigned<F>>, LEN, ReLu<F>>::without_witnesses(),
        //     l1: Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses(),
        //     //            l3: Nonlin1d::<F, Value<Assigned<F>>, LEN, ReLu<F>>::without_witnesses(),
        //     //            l4: Nonlin1d::<F, Value<Assigned<F>>, LEN, DivideBy<F, 128>>::without_witnesses(),
        // }
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
        let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

        // (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1 },
        //        { (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1 }

        let num_advices = max(max(output_width, IMAGE_WIDTH), LEN + 3);

        let advices = (0..num_advices)
            .map(|_| cs.advice_column())
            .collect::<Vec<_>>();
        for col in advices.iter() {
            cs.enable_equality(*col);
        }

        let l0 = cnvrl_generic::Config::<
            F,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        >::configure(cs, advices.clone());
        let l0q: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 32>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).clone().try_into().unwrap());
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).clone().try_into().unwrap());

        let l2: Affine1dConfig<F, LEN, CLASSES> = Affine1dConfig::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(),
            (&advices[LEN]).clone(),
            (&advices[CLASSES + 1]).clone(),
            (&advices[LEN + 2]).clone(),
        );
        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);
        //        cs.enable_equality(l2.output);

        MyConfig {
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
        let l0out = config
            .l0
            .assign(&mut layouter, self.input, self.l0_params)?;
        let l0qout = config.l0q.layout(
            &mut layouter,
            l0out.into_iter().flatten().flatten().collect(),
        )?;
        let l1out = config.l1.layout(&mut layouter, l0qout.clone())?;
        let l2out = config.l2.layout(
            &mut layouter,
            self.l2_params.0.clone(),
            self.l2_params.1.clone(),
            l1out.clone(),
        )?;
        //        println!("l1out {:?}", l1out);
        // println!(
        //     "L0Out: {:?}",
        //     l0out
        //         .iter()
        //         .map(|x| x.value_field().map(|y| felt_to_i32(y.evaluate())))
        //         .collect::<Vec<_>>()
        // );

        // println!(
        //     "L0qOut: {:?}",
        //     l0qout
        //         .iter()
        //         .map(|x| x.value_field().map(|y| felt_to_i32(y.evaluate())))
        //         .collect::<Vec<_>>()
        // );

        // println!(
        //     "L1Out: {:?}",
        //     l1out
        //         .iter()
        //         .map(|x| x.value_field().map(|y| felt_to_i32(y.evaluate())))
        //         .collect::<Vec<_>>()
        // );

        println!(
            "L2Out: {:?}",
            l2out
                .iter()
                .map(|x| x.value_field().map(|y| felt_to_i32(y.evaluate())))
                .collect::<Vec<_>>()
        );

        // tie the last output to public inputs (instance column)
        for i in 0..CLASSES {
            layouter.constrain_instance(l2out[i].cell(), config.public_output, i)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fieldutils::felt_to_i32;
    use halo2_proofs::{
        arithmetic::{Field, FieldExt},
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp as F,
        plonk::{Any, Circuit},
    };
    //     use nalgebra;
    use crate::cnvrl_generic::matrix;
    use crate::fieldutils;
    use crate::moreparams;
    use crate::tensorutils::map4;
    use halo2_proofs::pasta::pallas;
    use mnist::*;
    use ndarray::prelude::*;
    use rand::prelude::*;
    use rand::rngs::OsRng;
    use std::time::{Duration, Instant};

    const K: u32 = 17;

    const KERNEL_HEIGHT: usize = 5; //3
    const KERNEL_WIDTH: usize = 5; //3
    const OUT_CHANNELS: usize = 4;
    const STRIDE: usize = 2;
    const IMAGE_HEIGHT: usize = 28; //7
    const IMAGE_WIDTH: usize = 28; //7
    const IN_CHANNELS: usize = 1;
    const PADDING: usize = 0;
    const CLASSES: usize = 10;
    const LEN: usize = {
        ((IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
            * ((IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)
    };

    #[test]
    #[ignore]
    fn test_prove_mnist_inference() {
        // Load the parameters and preimage from somewhere

        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();

        let image_num = 0;
        // Can use an Array2 or Array3 here (Array3 for visualization)
        let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
            .expect("Error converting images to Array3 struct")
            //            .map(|x| *x as f32 / 256.0);
            .map(|x| i32tofelt::<F>(*x as i32 / 16));
        //        println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

        let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f32);
        println!(
            "The first digit is a {:?}",
            train_labels.slice(s![image_num, ..])
        );
        // let image = (0..IN_CHANNELS)
        //     .map(|i| {
        //         matrix(|| {
        //             Value::known({
        //                 i32tofelt::<F>(rand::random::<u8>() as i32)
        //                 //                        F::random(OsRng)
        //             })
        //         })
        //     })
        //     .collect::<Vec<_>>()
        //     .try_into()
        //     .unwrap();

        let mut image: [[[Value<F>; 28]; 28]; 1] = [[[Value::default(); 28]; 28]; 1];
        for i in 0..1 {
            for j in 0..28 {
                for k in 0..28 {
                    image[i][j][k] = Value::known(train_data[[image_num, j, k]]);
                }
            }
        }

        // let kernels = (0..OUT_CHANNELS)
        //     .map(|i| {
        //         (0..IN_CHANNELS)
        //             .map(|j| matrix(|| Value::known(i32tofelt(i as i32 - j as i32))))
        //             .collect::<Vec<_>>()
        //             .try_into()
        //             .unwrap()
        //     })
        //     .collect::<Vec<_>>()
        //     .try_into()
        //     .unwrap();

        let myparams = moreparams::Params::new();
        let tfkernels = map4::<Value<F>, _, KERNEL_HEIGHT, KERNEL_WIDTH, IN_CHANNELS, OUT_CHANNELS>(
            |i, j, k, l| {
                let dx = (myparams.kernels[i][j][k][l] as f32) * (32 as f32);
                let rounded = dx.round();
                let integral: i32 = unsafe { rounded.to_int_unchecked() };
                let felt = fieldutils::i32tofelt(integral);
                Value::known(felt)
            },
        );

        let l2biases: Vec<i32> = {
            let mut b = Vec::new();
            for fl in myparams.biases {
                let dx = fl * (32 as f32);
                let rounded = dx.round();
                let integral: i32 = unsafe { rounded.to_int_unchecked() };
                //                let felt = fieldutils::i32tofelt(integral);
                b.push(integral);
            }
            b
        };

        let tfweights: Vec<Vec<i32>> = {
            let mut w: Vec<Vec<i32>> = Vec::new();
            for row in myparams.weights {
                let mut newrow = Vec::new();
                for fl in row {
                    let dx = fl * (32 as f32);
                    let rounded = dx.round();
                    let integral: i32 = unsafe { rounded.to_int_unchecked() };
                    //                    let felt = fieldutils::i32tofelt(integral);
                    newrow.push(integral);
                }
                w.push(newrow);
            }
            w
        };

        let l2weights = {
            let mut w: Vec<Vec<i32>> = Vec::new();
            for i in 0..CLASSES {
                let mut row = Vec::new();
                for j in 0..LEN {
                    row.push(tfweights[j][i]);
                }
                w.push(row);
            }
            w
        };

        // tensorflow is in KHxKWxINxOUT we are OUTxINxWxH?
        let kernels: [[[[Value<F>; KERNEL_HEIGHT]; KERNEL_WIDTH]; IN_CHANNELS]; OUT_CHANNELS] = {
            let mut t4 =
                [[[[Value::unknown(); KERNEL_HEIGHT]; KERNEL_WIDTH]; IN_CHANNELS]; OUT_CHANNELS];
            for i in 0..OUT_CHANNELS {
                for j in 0..IN_CHANNELS {
                    for k in 0..KERNEL_WIDTH {
                        for l in 0..KERNEL_HEIGHT {
                            t4[i][j][k][l] = tfkernels[l][k][j][i];
                        }
                    }
                }
            }
            t4
        };

        // let l0weights: Vec<Vec<i32>> = vec![
        //     vec![10, 0, 0, -1],
        //     vec![0, 10, 1, 0],
        //     vec![0, 1, 10, 0],
        //     vec![1, 0, 0, 10],
        // ];
        // let l0biases: Vec<i32> = vec![0, 0, 0, 1];
        // let l2weights: Vec<Vec<i32>> = vec![vec![1; LEN]; LEN];
        // let l2biases: Vec<i32> = vec![1; LEN];

        let input = image; //: [cnvrl_generic::Image<Value<F>, IMAGE_HEIGHT, IMAGE_WIDTH>; IN_CHANNELS] = ;
        let l0_params = kernels; //         [[cnvrl_generic::Kernel<Value<F>, KERNEL_HEIGHT, KERNEL_WIDTH>; IN_CHANNELS]; OUT_CHANNELS] = ;
        let l2_params: (Vec<Vec<i32>>, Vec<i32>) = (l2weights, l2biases);

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

        let public_input = vec![
            -3283, -1071, -7182, -9264, -5729, 1197, -2673, -12619, -1283, -14700,
        ];

        // let nicer_pi: &[&[&[F]]] = &[&[&[
        //     i32tofelt::<F>(-9),
        //     i32tofelt::<F>(-15),
        //     i32tofelt::<F>(1),
        //     i32tofelt::<F>(-19),
        //     i32tofelt::<F>(7),
        //     i32tofelt::<F>(-18),
        //     i32tofelt::<F>(-6),
        //     i32tofelt::<F>(-8),
        //     i32tofelt::<F>(-19),
        // ]]];

        let pi_inner: [F; CLASSES] = public_input
            .iter()
            .map(|x| i32tofelt::<F>(*x).into())
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner]];

        //        Mock Proof
        // let prover = MockProver::run(
        //     K,
        //     &circuit,
        //     vec![public_input
        //         .iter()
        //         .map(|x| i32tofelt::<F>(*x).into())
        //         .collect()],
        //     //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
        // )
        // .unwrap();
        // prover.assert_satisfied();

        //	Real proof
        let params: Params<vesta::Affine> = Params::new(K);
        let empty_circuit = circuit.without_witnesses();
        // Initialize the proving key
        let now = Instant::now();
        let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
        println!("VK took {}", now.elapsed().as_secs());
        let now = Instant::now();
        let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
        println!("PK took {}", now.elapsed().as_secs());
        let now = Instant::now();
        //println!("{:?} {:?}", vk, pk);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let mut rng = OsRng;
        create_proof(
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
        let strategy = SingleVerifier::new(&params);
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

    #[cfg(feature = "dev-graph")]
    #[test]
    fn print_convrelaffrel() {
        use plotters::prelude::*;

        let root = BitMapBackend::new("convrelaffrel-layout.png", (2048, 7680)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Conv -> ReLu -> Affine -> Relu", ("sans-serif", 60))
            .unwrap();

        let circuit = MyCircuit::<
            F,
            LEN,
            10,
            14,
            14,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        > {
            input: [[[Value::unknown(); IMAGE_HEIGHT]; IMAGE_WIDTH]; IN_CHANNELS],
            l0_params: [[[[Value::unknown(); KERNEL_HEIGHT]; KERNEL_WIDTH]; IN_CHANNELS];
                OUT_CHANNELS],
            l2_params: (vec![vec![1; LEN]; LEN], vec![1; LEN]),
        };
        halo2_proofs::dev::CircuitLayout::default()
            .render(13, &circuit, &root)
            .unwrap();
    }
}

//

// [Value { inner: Some(0) }, Value { inner: Some(3) }, Value { inner: Some(10) }, Value { inner: Some(-1) }, Value { inner: Some(16) }, Value { inner: Some(0) }, Value { inner: Some(3) }, Value { inner: Some(10) }, Value { inner: Some(-1) }]
