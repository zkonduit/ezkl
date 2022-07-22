use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
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

use crate::another1d::{Affine1d, Affine1dConfig};
use crate::cnvrl_generic;
use crate::eltwise::{DivideBy, Nonlin1d, NonlinConfig1d, ReLu};

#[derive(Clone)]
struct MyConfig<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust stable
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
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, LEN>,
    //    l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    // l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
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
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).clone().try_into().unwrap());

        let l2: Affine1dConfig<F, LEN, LEN> = Affine1dConfig::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(),
            (&advices[LEN]).clone(),
            (&advices[LEN + 1]).clone(),
            (&advices[LEN + 2]).clone(),
        );
        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);
        cs.enable_equality(l2.output);

        MyConfig {
            l0,
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
        let l1out = config.l1.layout(
            &mut layouter,
            l0out.into_iter().flatten().flatten().collect(),
        )?;
        let l2out = config.l2.layout(
            &mut layouter,
            self.l2_params.0.clone(),
            self.l2_params.1.clone(),
            l1out.clone(),
        )?;
        //        println!("l1out {:?}", l1out);
        println!(
            "{:?}",
            l2out
                .iter()
                .map(|x| x.value_field().map(|y| felt_to_i32(y.evaluate())))
                .collect::<Vec<_>>()
        );

        // tie the last output to public inputs (instance column)
        for i in 0..LEN {
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
    use halo2_proofs::pasta::pallas;
    use rand::prelude::*;
    use rand::rngs::OsRng;
    use std::time::{Duration, Instant};

    #[test]
    fn test_convrelaffrel() {
        let k = 15; //2^k rows
                    // parameters

        const KERNEL_HEIGHT: usize = 2; //3
        const KERNEL_WIDTH: usize = 2; //3
        const OUT_CHANNELS: usize = 2;
        const STRIDE: usize = 2;
        const IMAGE_HEIGHT: usize = 3; //7
        const IMAGE_WIDTH: usize = 3; //7
        const IN_CHANNELS: usize = 2;
        const PADDING: usize = 2;

        const LEN: usize = {
            ((IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
                * ((IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)
        };
        // [(); (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1]:,
        // [(); (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1]:,
        //5x5?

        // Load the parameters and preimage from somewhere

        let image = (0..IN_CHANNELS)
            .map(|i| matrix(|| Value::known(i32tofelt(i as i32))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let kernels = (0..OUT_CHANNELS)
            .map(|i| {
                (0..IN_CHANNELS)
                    .map(|j| matrix(|| Value::known(i32tofelt(i as i32 - j as i32))))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // let l0weights: Vec<Vec<i32>> = vec![
        //     vec![10, 0, 0, -1],
        //     vec![0, 10, 1, 0],
        //     vec![0, 1, 10, 0],
        //     vec![1, 0, 0, 10],
        // ];
        // let l0biases: Vec<i32> = vec![0, 0, 0, 1];
        let l2weights: Vec<Vec<i32>> = vec![
            vec![1; 9],
            vec![2; 9],
            vec![1; 9],
            vec![2; 9],
            vec![1; 9],
            vec![2; 9],
            vec![1; 9],
            vec![2; 9],
            vec![2; 9],
        ];

        let l2biases: Vec<i32> = vec![
            0, 3, 10, -1, 16, 0, 3, 10, -1, 16, 0, 3, 10, -1, 16, 0, 3, 10, -1, 16, 0, 3, 10, -1,
            16,
        ];

        let input = image; //: [cnvrl_generic::Image<Value<F>, IMAGE_HEIGHT, IMAGE_WIDTH>; IN_CHANNELS] = ;
        let l0_params = kernels; //         [[cnvrl_generic::Kernel<Value<F>, KERNEL_HEIGHT, KERNEL_WIDTH>; IN_CHANNELS]; OUT_CHANNELS] = ;
        let l2_params: (Vec<Vec<i32>>, Vec<i32>) = (l2weights, l2biases);

        let circuit = MyCircuit::<
            F,
            LEN,
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
            input,
            l0_params,
            l2_params,
        };

        let public_input = vec![0, 3, 10, -1, 16, 0, 3, 10, -1];

        //[-9, -15, 1, -19, 7, -18, -6, -8, -19]

        // let public_input: Vec<i32> = unsafe {
        //     vec![
        //         (531f32 / 128f32).round().to_int_unchecked::<i32>().into(),
        //         (103f32 / 128f32).round().to_int_unchecked::<i32>().into(),
        //         (4469f32 / 128f32).round().to_int_unchecked::<i32>().into(),
        //         (2849f32 / 128f32).to_int_unchecked::<i32>().into(),
        //     ]
        // };

        println!("public input {:?}", public_input);

        let nicer_pi: &[&[&[F]]] = &[&[&[
            i32tofelt::<F>(-9),
            i32tofelt::<F>(-15),
            i32tofelt::<F>(1),
            i32tofelt::<F>(-19),
            i32tofelt::<F>(7),
            i32tofelt::<F>(-18),
            i32tofelt::<F>(-6),
            i32tofelt::<F>(-8),
            i32tofelt::<F>(-19),
        ]]];

        // vec![public_input
        // .iter()
        // .map(|x| i32tofelt::<F>(*x).into())
        // .collect()];

        let prover = MockProver::run(
            k,
            &circuit,
            vec![public_input
                .iter()
                .map(|x| i32tofelt::<F>(*x).into())
                .collect()],
            //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
        )
        .unwrap();
        prover.assert_satisfied();

        // let params: Params<vesta::Affine> = Params::new(k);

        // let empty_circuit = circuit.without_witnesses();

        // // Initialize the proving key
        // let now = Instant::now();
        // let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
        // println!("VK took {}", now.elapsed().as_secs());
        // let now = Instant::now();
        // let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
        // println!("PK took {}", now.elapsed().as_secs());
        // let now = Instant::now();
        // //println!("{:?} {:?}", vk, pk);
        // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        // let mut rng = OsRng;
        // create_proof(
        //     &params,
        //     &pk,
        //     &[circuit],
        //     nicer_pi,
        //     &mut rng,
        //     &mut transcript,
        // )
        // .expect("proof generation should not fail");
        // let proof = transcript.finalize();
        // //println!("{:?}", proof);
        // println!("Proof took {}", now.elapsed().as_secs());
        // let now = Instant::now();
        // let strategy = SingleVerifier::new(&params);
        // let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        // assert!(verify_proof(&params, pk.get_vk(), strategy, nicer_pi, &mut transcript).is_ok());
        // println!("Verify took {}", now.elapsed().as_secs());
    }
}

//

// [Value { inner: Some(0) }, Value { inner: Some(3) }, Value { inner: Some(10) }, Value { inner: Some(-1) }, Value { inner: Some(16) }, Value { inner: Some(0) }, Value { inner: Some(3) }, Value { inner: Some(10) }, Value { inner: Some(-1) }]
